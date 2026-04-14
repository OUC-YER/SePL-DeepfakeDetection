# author: Enrui Yang
# email: yer3888@stu.ouc.edu.cn
# date:2026-0331
# description: SePL_Detector

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

from transformers import CLIPModel

from networks.svd_linear import apply_svd_residual_to_self_attn, SVDResidualLinear
from loss.sepl_loss import SupConLoss, MultiModalAlignmentLoss
from detectors.sepl_modules import GuidedDecoupling

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='sepl')
class SePLDetector(nn.Module):
    def __init__(self, config=None):
        super(SePLDetector, self).__init__()
        self.config = config

        self.gate_init_bias = self._get_cfg_float('gate_init_bias', 3.0)
        self.backbone_dim = int(self._get_cfg_float('backbone_dim', 1024))
        self.content_dim = int(self._get_cfg_float('content_dim', 512))
        self.artifact_dim = int(self._get_cfg_float('artifact_dim', 512))
        self.n_ctx = int(self._get_cfg_float('n_ctx', 16))
        self.meta_net_hidden_dim = int(self._get_cfg_float('meta_net_hidden_dim', 256))

        self.lambda_contrast = self._get_cfg_float('lambda_contrast', 0.1)
        self.lambda_decouple = self._get_cfg_float('lambda_decouple', 0.05)
        self.lambda_content_align = self._get_cfg_float('lambda_content_align', 0.08)
        self.lambda_artifact_align = self._get_cfg_float('lambda_artifact_align', 0.12)
        self.lambda_prompt_diversity = self._get_cfg_float('lambda_prompt_diversity', 0.01)

        self.clip_path = "/data/disk2/yer/ASOTA/DeepfakeBench/training/config/vit/models--openai--clip-vit-large-patch14"

        self.clip_model = CLIPModel.from_pretrained(self.clip_path)
        r = int(self._get_cfg_float('svd_r', 1023))
        self.clip_model.vision_model = apply_svd_residual_to_self_attn(
            self.clip_model.vision_model, r=r, gate_init_bias=self.gate_init_bias
        )
        self.backbone = self.clip_model.vision_model

        self.decouple = GuidedDecoupling(
            clip_model=self.clip_model,
            image_dim=self.backbone_dim,
            content_dim=self.content_dim,
            artifact_dim=self.artifact_dim,
            n_ctx=self.n_ctx,
            use_cross_attention=True,
            meta_net_hidden_dim=self.meta_net_hidden_dim
        )

        self.head = nn.Linear(self.artifact_dim, 2)

        self.contrast_loss_fn = SupConLoss(temperature=0.07)
        self.alignment_loss_fn = MultiModalAlignmentLoss(temperature=0.07, projection_dim=self.content_dim)
        self.loss_func = nn.CrossEntropyLoss()

        self._train_step = 0
        self.total_steps = self._get_cfg_float('total_steps', 20000)

        self.best_video_auc = 0.0
        self.patience_counter = 0
        self.patience = int(self._get_cfg_float('patience', 15))

    def _get_cfg_float(self, key, default):
        if self.config is None:
            return default
        if hasattr(self.config, key):
            try:
                return getattr(self.config, key)
            except Exception:
                return default
        if isinstance(self.config, dict) and key in self.config:
            return self.config[key]
        return default

    def setup_pretrain(self):
        device = next(self.parameters()).device

        self._saved_requires_grad = {}
        for name, p in self.named_parameters():
            self._saved_requires_grad[name] = p.requires_grad

        logger.info("[Pretrain] Loading frozen CLIP model for content meta-net pretraining...")
        self.pretrain_clip = CLIPModel.from_pretrained(self.clip_path).to(device)
        self.pretrain_clip.eval()
        for p in self.pretrain_clip.parameters():
            p.requires_grad = False

        for p in self.parameters():
            p.requires_grad = False

        for p in self.decouple.prompt_learner.meta_net_content.parameters():
            p.requires_grad = True

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"[Pretrain] Trainable params: {trainable_params:,}")

    def pretrain_content_step(self, data_dict):
        img = data_dict['image']
        batch_size = img.size(0)
        device = img.device

        with torch.no_grad():
            vision_outputs = self.pretrain_clip.vision_model(img)
            image_features = vision_outputs.pooler_output

        conditional_token = self.decouple.prompt_learner.meta_net_content(image_features)
        conditional_token = conditional_token.unsqueeze(1)

        ctx = self.decouple.prompt_learner.ctx_content.unsqueeze(0).expand(batch_size, -1, -1)
        content_prompts = torch.cat([conditional_token, ctx], dim=1)

        text_features = self.decouple.encode_text_prompts(content_prompts)

        with torch.no_grad():
            image_embeds = self.pretrain_clip.visual_projection(image_features)
            image_embeds = F.normalize(image_embeds, dim=-1)

        text_embeds = self.pretrain_clip.text_projection(text_features)
        text_embeds = F.normalize(text_embeds, dim=-1)

        logit_scale = self.pretrain_clip.logit_scale.exp()
        logits = image_embeds @ text_embeds.T * logit_scale

        labels = torch.arange(batch_size, device=device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        loss = (loss_i2t + loss_t2i) / 2.0

        return loss

    def cleanup_pretrain(self):
        if hasattr(self, 'pretrain_clip'):
            del self.pretrain_clip
            torch.cuda.empty_cache()
            logger.info("[Pretrain] Freed pretrain CLIP model memory.")

        if hasattr(self, '_saved_requires_grad'):
            for name, p in self.named_parameters():
                if name in self._saved_requires_grad:
                    p.requires_grad = self._saved_requires_grad[name]
            del self._saved_requires_grad
            logger.info("[Pretrain] Restored all parameter requires_grad states.")

    def compute_prompt_diversity_loss(self, text_features):
        text_norm = F.normalize(text_features, dim=1)
        similarity = torch.matmul(text_norm, text_norm.T)
        mask = torch.eye(similarity.size(0), device=similarity.device).bool()
        similarity = similarity.masked_fill(mask, 0)
        return similarity.abs().mean()

    def features(self, data_dict: dict) -> torch.Tensor:
        img = data_dict['image']
        outputs = self.backbone(img)
        if isinstance(outputs, dict) or hasattr(outputs, 'get'):
            feat_backbone = outputs['pooler_output']
        else:
            feat_backbone = outputs.pooler_output
        return feat_backbone

    def classifier(self, artifact_feat: torch.Tensor) -> torch.Tensor:
        return self.head(artifact_feat)

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)

        mask_real = label == 0
        mask_fake = label == 1
        loss_real = self.loss_func(pred[mask_real], label[mask_real]) if mask_real.sum() > 0 else torch.tensor(0.0, device=pred.device)
        loss_fake = self.loss_func(pred[mask_fake], label[mask_fake]) if mask_fake.sum() > 0 else torch.tensor(0.0, device=pred.device)

        if self.training:
            self._train_step += 1
            lambdas = self._get_dynamic_lambdas(self._train_step, self.total_steps)

            orth_loss = 0.0
            ksv_loss = 0.0
            num_reg = 0
            for module in self.backbone.modules():
                if isinstance(module, SVDResidualLinear):
                    orth_loss += module.compute_orthogonal_loss()
                    ksv_loss += module.compute_keepsv_loss()
                    num_reg += 1
            if num_reg > 0:
                orth_loss /= num_reg
                ksv_loss /= num_reg

            feat_backbone = pred_dict['feat_backbone']
            contrast_loss = self.contrast_loss_fn(feat_backbone, label)

            content_feat = pred_dict['content_feat']
            artifact_feat = pred_dict['artifact_feat']
            decouple_loss = self.decouple.compute_orthogonal_loss(content_feat, artifact_feat)

            content_text_feat = pred_dict['content_text_feat']
            artifact_text_feat = pred_dict['artifact_text_feat']
            content_align_loss = self.alignment_loss_fn(content_feat, content_text_feat, mode='content')
            artifact_align_loss = self.alignment_loss_fn(artifact_feat, artifact_text_feat, labels=label, mode='artifact')

            prompt_diversity_loss = (
                self.compute_prompt_diversity_loss(content_text_feat) +
                self.compute_prompt_diversity_loss(artifact_text_feat)
            ) / 2.0

            overall_loss = (
                loss +
                lambdas['lambda_orth'] * orth_loss +
                lambdas['lambda_ksv'] * ksv_loss +
                lambdas['lambda_contrast'] * contrast_loss +
                lambdas['lambda_decouple'] * decouple_loss +
                lambdas['lambda_content_align'] * content_align_loss +
                lambdas['lambda_artifact_align'] * artifact_align_loss +
                self.lambda_prompt_diversity * prompt_diversity_loss
            )
        else:
            overall_loss = loss
            orth_loss = torch.tensor(0.0, device=pred.device)
            ksv_loss = torch.tensor(0.0, device=pred.device)
            contrast_loss = torch.tensor(0.0, device=pred.device)
            decouple_loss = torch.tensor(0.0, device=pred.device)
            content_align_loss = torch.tensor(0.0, device=pred.device)
            artifact_align_loss = torch.tensor(0.0, device=pred.device)
            prompt_diversity_loss = torch.tensor(0.0, device=pred.device)

        loss_dict = {
            'overall': overall_loss,
            'cls_loss': loss,
            'real_loss': loss_real,
            'fake_loss': loss_fake,
            'orth_loss': orth_loss,
            'ksv_loss': ksv_loss,
            'contrast_loss': contrast_loss,
            'decouple_loss': decouple_loss,
            'content_align_loss': content_align_loss,
            'artifact_align_loss': artifact_align_loss,
            'prompt_diversity_loss': prompt_diversity_loss,
        }
        return loss_dict

    def _get_dynamic_lambdas(self, current_step, total_steps):
        warmup_ratio = 0.1
        warmup_steps = int(total_steps * warmup_ratio)

        if current_step < warmup_steps:
            factor = current_step / warmup_steps
            lambda_orth = 0.02 * factor
            lambda_ksv = 0.03 * factor
            lambda_contrast = self.lambda_contrast * factor
            lambda_decouple = self.lambda_decouple * factor
            lambda_content_align = self.lambda_content_align * factor
            lambda_artifact_align = self.lambda_artifact_align * factor
        else:
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            decay_factor = 1.0 if progress < 0.5 else 1.0 - 0.3 * (progress - 0.5) / 0.5

            lambda_orth = 0.02 * decay_factor
            lambda_ksv = 0.03 * decay_factor
            lambda_contrast = self.lambda_contrast
            lambda_decouple = self.lambda_decouple
            lambda_content_align = self.lambda_content_align
            lambda_artifact_align = self.lambda_artifact_align

        return {
            'lambda_orth': lambda_orth,
            'lambda_ksv': lambda_ksv,
            'lambda_contrast': lambda_contrast,
            'lambda_decouple': lambda_decouple,
            'lambda_content_align': lambda_content_align,
            'lambda_artifact_align': lambda_artifact_align,
        }

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

    def forward(self, data_dict: dict, inference=False) -> dict:
        feat_backbone = self.features(data_dict)

        content_feat, artifact_feat, content_text_feat, artifact_text_feat = self.decouple(feat_backbone)

        pred = self.classifier(artifact_feat)
        prob = torch.softmax(pred, dim=1)[:, 1]

        pred_dict = {
            'cls': pred,
            'prob': prob,
            'feat': artifact_feat,
            'feat_backbone': feat_backbone,
            'content_feat': content_feat,
            'artifact_feat': artifact_feat,
            'content_text_feat': content_text_feat,
            'artifact_text_feat': artifact_text_feat,
        }
        return pred_dict
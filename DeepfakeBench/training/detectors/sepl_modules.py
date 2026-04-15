# author: Enrui Yang
# email: yer3888@stu.ouc.edu.cn
# date:2026-0331
# description: SePL_Modules

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalPromptLearner(nn.Module):
    def __init__(self, clip_model, image_encoder_dim=1024, n_ctx=16,
                 class_specific=True, meta_net_hidden_dim=256):
        super(ConditionalPromptLearner, self).__init__()
        self.n_ctx = n_ctx
        self.class_specific = class_specific

        dtype = clip_model.text_model.embeddings.token_embedding.weight.dtype
        self.dtype = dtype
        ctx_dim = clip_model.text_model.config.hidden_size

        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)

        if class_specific:
            self.ctx_forgery_irrelevant = nn.Parameter(ctx_vectors.clone())
            self.ctx_forgery_specific = nn.Parameter(ctx_vectors.clone())
            print(f"Initialized class-specific static contexts: {n_ctx} tokens")
        else:
            self.ctx = nn.Parameter(ctx_vectors)
            print(f" Initialized shared static contexts: {n_ctx} tokens")

        if class_specific:
            self.meta_net_forgery_irrelevant = nn.Sequential(
                nn.Linear(image_encoder_dim, meta_net_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(meta_net_hidden_dim, ctx_dim)
            )
            self.meta_net_forgery_specific = nn.Sequential(
                nn.Linear(image_encoder_dim, meta_net_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(meta_net_hidden_dim, ctx_dim)
            )
            print(f"[] Built class-specific meta-networks: {image_encoder_dim} -> {ctx_dim}")
        else:
            self.meta_net = nn.Sequential(
                nn.Linear(image_encoder_dim, meta_net_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(meta_net_hidden_dim, ctx_dim)
            )

        self.clip_model = clip_model

    def forward(self, image_features):
        batch_size = image_features.size(0)

        if self.class_specific:
            ctx_forgery_irrelevant = self.ctx_forgery_irrelevant.unsqueeze(0).expand(batch_size, -1, -1)
            ctx_forgery_specific = self.ctx_forgery_specific.unsqueeze(0).expand(batch_size, -1, -1)

            conditional_forgery_irrelevant = self.meta_net_forgery_irrelevant(image_features)
            conditional_forgery_specific = self.meta_net_forgery_specific(image_features)

            conditional_forgery_irrelevant = conditional_forgery_irrelevant.unsqueeze(1)
            conditional_forgery_specific = conditional_forgery_specific.unsqueeze(1)
        else:
            ctx = self.ctx.unsqueeze(0).expand(batch_size, -1, -1)
            conditional = self.meta_net(image_features).unsqueeze(1)
            ctx_forgery_irrelevant = ctx
            ctx_forgery_specific = ctx
            conditional_forgery_irrelevant = conditional
            conditional_forgery_specific = conditional

        forgery_irrelevant_prompts = torch.cat([conditional_forgery_irrelevant, ctx_forgery_irrelevant], dim=1)
        forgery_specific_prompts = torch.cat([conditional_forgery_specific, ctx_forgery_specific], dim=1)

        return forgery_irrelevant_prompts, forgery_specific_prompts


class TextGuidedFeatureEncoder(nn.Module):
    def __init__(self, image_dim=1024, text_dim=768, output_dim=512, use_cross_attention=True):
        super(TextGuidedFeatureEncoder, self).__init__()
        self.use_cross_attention = use_cross_attention

        if use_cross_attention:
            self.text_dim_proj = nn.Linear(text_dim, image_dim) if text_dim != image_dim else nn.Identity()

            self.cross_attn = nn.MultiheadAttention(
                embed_dim=image_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            self.norm1 = nn.LayerNorm(image_dim)

            self.ffn = nn.Sequential(
                nn.Linear(image_dim, image_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(image_dim * 2, output_dim),
                nn.LayerNorm(output_dim)
            )
        else:
            self.image_proj = nn.Linear(image_dim, output_dim)
            self.text_proj = nn.Linear(text_dim, output_dim)
            self.fusion = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.GELU(),
                nn.LayerNorm(output_dim)
            )

    def forward(self, image_feat, text_feat):
        if self.use_cross_attention:
            image_feat_expanded = image_feat.unsqueeze(1)
            text_feat_expanded = self.text_dim_proj(text_feat).unsqueeze(1)

            attn_out, _ = self.cross_attn(
                query=image_feat_expanded,
                key=text_feat_expanded,
                value=text_feat_expanded
            )
            attn_out = self.norm1(attn_out + image_feat_expanded)
            guided_feat = self.ffn(attn_out.squeeze(1))
        else:
            img_proj = self.image_proj(image_feat)
            txt_proj = self.text_proj(text_feat)
            guided_feat = self.fusion(torch.cat([img_proj, txt_proj], dim=-1))

        return guided_feat


class GuidedDecoupling(nn.Module):
    def __init__(self, clip_model, image_dim=1024, forgery_irrelevant_dim=512, forgery_specific_dim=512,
                 n_ctx=16, use_cross_attention=True, meta_net_hidden_dim=256):
        super(GuidedDecoupling, self).__init__()

        self.clip_model = clip_model
        text_dim = clip_model.text_model.config.hidden_size

        self.prompt_learner = ConditionalPromptLearner(
            clip_model=clip_model,
            image_encoder_dim=image_dim,
            n_ctx=n_ctx,
            class_specific=True,
            meta_net_hidden_dim=meta_net_hidden_dim
        )

        self.forgery_irrelevant_encoder = TextGuidedFeatureEncoder(
            image_dim=image_dim,
            text_dim=text_dim,
            output_dim=forgery_irrelevant_dim,
            use_cross_attention=use_cross_attention
        )

        self.forgery_specific_encoder = TextGuidedFeatureEncoder(
            image_dim=image_dim,
            text_dim=text_dim,
            output_dim=forgery_specific_dim,
            use_cross_attention=use_cross_attention
        )

    def encode_text_prompts(self, prompt_embeddings):
        text_model = self.clip_model.text_model
        device = prompt_embeddings.device
        batch_size = prompt_embeddings.size(0)
        seq_len = prompt_embeddings.size(1)

        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, -1)
        position_embeddings = text_model.embeddings.position_embedding(position_ids)

        embeddings = prompt_embeddings + position_embeddings

        attention_mask = torch.zeros(
            (batch_size, 1, seq_len, seq_len),
            dtype=embeddings.dtype,
            device=device
        )
        causal_mask = torch.zeros(
            (batch_size, 1, seq_len, seq_len),
            dtype=embeddings.dtype,
            device=device
        )

        hidden_states = embeddings
        for layer in text_model.encoder.layers:
            layer_output = layer(hidden_states, attention_mask, causal_mask)
            hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output

        hidden_states = text_model.final_layer_norm(hidden_states)
        text_features = hidden_states[:, -1, :]

        return text_features

    def forward(self, image_feat):
        forgery_irrelevant_prompts, forgery_specific_prompts = self.prompt_learner(image_feat)

        forgery_irrelevant_text_feat = self.encode_text_prompts(forgery_irrelevant_prompts)
        forgery_specific_text_feat = self.encode_text_prompts(forgery_specific_prompts)

        forgery_irrelevant_feat = self.forgery_irrelevant_encoder(image_feat, forgery_irrelevant_text_feat)
        forgery_specific_feat = self.forgery_specific_encoder(image_feat, forgery_specific_text_feat)

        return forgery_irrelevant_feat, forgery_specific_feat, forgery_irrelevant_text_feat, forgery_specific_text_feat

    def compute_orthogonal_loss(self, forgery_irrelevant_feat, forgery_specific_feat):
        forgery_irrelevant_norm = F.normalize(forgery_irrelevant_feat, dim=1)
        forgery_specific_norm = F.normalize(forgery_specific_feat, dim=1)
        correlation = torch.abs(torch.sum(forgery_irrelevant_norm * forgery_specific_norm, dim=1))
        return correlation.mean()
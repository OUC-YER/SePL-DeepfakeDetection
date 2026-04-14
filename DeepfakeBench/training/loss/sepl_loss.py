# author: Enrui Yang
# email: yer3888@stu.ouc.edu.cn
# date:2026-0331
# description: SePL_Loss

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        return loss


class MultiModalAlignmentLoss(nn.Module):
    def __init__(self, temperature=0.07, projection_dim=512):
        super(MultiModalAlignmentLoss, self).__init__()
        self.temperature = temperature

        self.text_projector = nn.Sequential(
            nn.Linear(768, projection_dim),
            nn.LayerNorm(projection_dim)
        )

    def forward(self, image_feat, text_feat, labels=None, mode='content'):
        text_feat_projected = self.text_projector(text_feat)

        image_feat_norm = F.normalize(image_feat, dim=1)
        text_feat_norm = F.normalize(text_feat_projected, dim=1)

        if mode == 'content':
            similarity = torch.sum(image_feat_norm * text_feat_norm, dim=1)
            loss = -similarity.mean()

        elif mode == 'artifact':
            if labels is None:
                similarity = torch.sum(image_feat_norm * text_feat_norm, dim=1)
                loss = -similarity.mean()
            else:
                similarity = torch.sum(image_feat_norm * text_feat_norm, dim=1)
                target_similarity = labels.float() * 2 - 1
                loss = -torch.mean(similarity * target_similarity)

        return loss
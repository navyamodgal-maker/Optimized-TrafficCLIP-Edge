import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """Supervised contrastive loss on fused visual features (Eq. 9 in the paper)."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature

        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        self_mask = torch.eye(mask.shape[0], device=device)
        mask = mask - self_mask

        exp_sim = torch.exp(sim) * (1 - self_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        pos_count = mask.sum(dim=1)
        pos_count = torch.clamp(pos_count, min=1)
        loss = -(mask * log_prob).sum(dim=1) / pos_count
        return loss.mean()
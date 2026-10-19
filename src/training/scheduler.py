import math
import torch

def get_scheduler(optimizer, total_epochs, warmup_epochs=1, base_lr=0.002, warmup_lr=1e-5):
    """Cosine annealing with a 1-epoch warm-up, matching the paper's setup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return warmup_lr / base_lr
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
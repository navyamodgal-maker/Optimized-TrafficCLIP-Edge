import torch.nn as nn

class TrafficCrossEntropy(nn.Module):
    """Cross-entropy loss over the similarity logits (Eq. 8 in the paper)."""
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        return self.ce(logits, labels)
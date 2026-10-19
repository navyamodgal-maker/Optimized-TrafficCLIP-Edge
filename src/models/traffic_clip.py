import torch
import torch.nn as nn

from src.models.detail_encoder import DetailAwareEncoder
from src.models.semantic_encoder import SemanticEncoder
from src.models.adapter import TrafficAdapter
from src.models.fusion import FusionModule
from src.models.text_encoder import TextEncoder

class TrafficCLIP(nn.Module):
    def __init__(self, embed_dim=1024):
        super().__init__()
        self.detail_encoder = DetailAwareEncoder()
        self.semantic_encoder = SemanticEncoder()
        self.adapter = TrafficAdapter()
        self.fusion = FusionModule(output_dim=embed_dim)
        self.text_encoder = TextEncoder(output_dim=embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def forward(self, images, texts):
        detail_feat = self.detail_encoder(images)
        semantic_feat = self.adapter(self.semantic_encoder(images))
        visual_feat = self.fusion(detail_feat, semantic_feat)
        text_feat = self.text_encoder(texts)

        visual_feat = visual_feat / visual_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * visual_feat @ text_feat.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text

if __name__ == "__main__":
    model = TrafficCLIP()
    images = torch.randn(2, 1, 28, 28)
    texts = ["A network traffic photo of Skype", "A network traffic photo of Zoom"]
    logits_per_image, logits_per_text = model(images, texts)
    print(logits_per_image.shape)  # expect torch.Size([2, 2])
import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, detail_dim=512, semantic_dim=512, output_dim=1024):
        super().__init__()
        combined_dim = detail_dim + semantic_dim
        self.proj = nn.Identity() if combined_dim == output_dim else nn.Linear(combined_dim, output_dim)

    def forward(self, detail_feat, semantic_feat):
        fused = torch.cat([detail_feat, semantic_feat], dim=1)
        return self.proj(fused)

if __name__ == "__main__":
    fusion = FusionModule()
    detail_feat = torch.randn(2, 512)
    semantic_feat = torch.randn(2, 512)
    out = fusion(detail_feat, semantic_feat)
    print(out.shape)  # expect torch.Size([2, 1024])
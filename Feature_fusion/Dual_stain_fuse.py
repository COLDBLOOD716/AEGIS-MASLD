import os
import torch
import torch.nn as nn
from tqdm import tqdm

class Config:
    he_dir = r"path_to_HE_features"
    sr_dir = r"path_to_SR_features"
    output_dir = r"path_to_fused_output"
    fused_dim = 1280
    device = 'cuda'

os.makedirs(Config.output_dir, exist_ok=True)

class CrossModalFusion(nn.Module):
    def __init__(self, dim=768, num_heads=8, fused_dim=1280):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(2 * dim, fused_dim),
            nn.ReLU(),
            nn.LayerNorm(fused_dim)
        )
        self.contribution_proj = nn.Linear(dim, 1, bias=False)

    def forward(self, he_feat, sr_feat):
        q = self.query(he_feat.unsqueeze(1))
        k = self.key(sr_feat.unsqueeze(1))
        v = self.value(sr_feat.unsqueeze(1))
        attn_output, _ = self.mha(query=q, key=k, value=v)
        residual = torch.cat([attn_output.squeeze(1), he_feat], dim=1)
        return self.proj(residual)

    def get_contributions(self, he_feat, sr_feat):
        alpha = torch.sigmoid(self.contribution_proj(he_feat))
        beta = torch.sigmoid(self.contribution_proj(sr_feat))
        return alpha.squeeze(), beta.squeeze()

model = CrossModalFusion(fused_dim=Config.fused_dim).to(Config.device)
model.eval()

contrib_csv = os.path.join(Config.output_dir, 'contribution_records.csv')
with open(contrib_csv, 'w') as f:
    f.write("slide_id,he_contribution,sr_contribution\n")

he_files = [f for f in os.listdir(Config.he_dir) if f.endswith('-HE_embed.pt')]
pbar = tqdm(he_files, desc="Fusing features")

for he_file in pbar:
    try:
        sr_file = he_file.replace('-HE_embed.pt', '-SR_embed.pt')
        sr_path = os.path.join(Config.sr_dir, sr_file)
        if not os.path.exists(sr_path):
            continue

        he_feat = torch.load(os.path.join(Config.he_dir, he_file)).to(Config.device)
        sr_feat = torch.load(sr_path).to(Config.device)

        assert he_feat.shape == (1, 768)
        assert sr_feat.shape == (1, 768)

        with torch.no_grad():
            fused_feat = model(he_feat, sr_feat)
            alpha, beta = model.get_contributions(he_feat, sr_feat)

        slide_id = he_file.replace('-HE_embed.pt', '')
        with open(contrib_csv, 'a') as f:
            f.write(f"{slide_id},{alpha.item():.4f},{beta.item():.4f}\n")

        output_path = os.path.join(Config.output_dir, he_file.replace('-HE_embed.pt', '-Fused_embed.pt'))
        torch.save(fused_feat.cpu(), output_path)
        pbar.set_postfix({'current': he_file, 'fused_dim': fused_feat.shape[-1]})

    except Exception as e:
        continue

print("Dual feature fusion completed.")

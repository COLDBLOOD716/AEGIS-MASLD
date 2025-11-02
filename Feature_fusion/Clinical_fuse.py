import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

CLINICAL_CSV = r"path_to_clinical_csv"
FUSED_EMBED_DIR = r"path_to_fused_embeddings"
OUTPUT_DIR = r"path_to_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BINARY_COLS = ["HBsAg", "HBsAb", "HBeAg", "HBeAb", "HBcAb"]
CONT_COLS = ["ALT", "AST", "GGT", "TBIL", "DBIL", "Creatinine",
             "Uric Acid", "Blood glucose", "TG", "TC", "HDL", "LDL"]

df = pd.read_csv(CLINICAL_CSV, dtype={"ID": str})
sub = df[["ID"] + BINARY_COLS + CONT_COLS].set_index("ID")

bin_data = sub[BINARY_COLS].astype(float)

scaler = StandardScaler()
cont_data = pd.DataFrame(
    scaler.fit_transform(sub[CONT_COLS]),
    index=sub.index,
    columns=CONT_COLS
)

clinical_feats = pd.concat([bin_data, cont_data], axis=1)
ids = clinical_feats.index.tolist()
X_clin = torch.tensor(clinical_feats.values, dtype=torch.float)

class StaticImageFusion(nn.Module):
    def __init__(self, dim_clin, dim_img, fused_dim=1024, hidden_dim=1024, nhead=8, num_layers=1):
        super().__init__()
        self.fused_dim = fused_dim
        self.clin_proj = nn.Linear(dim_clin, fused_dim)
        self.img_proj = nn.Linear(dim_img, fused_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fused_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(fused_dim)

    def forward(self, clin_feat, img_feat):
        q = self.clin_proj(clin_feat)
        k = self.img_proj(img_feat)
        x = torch.cat([q, k], dim=1)
        x_out = self.transformer(x)
        fused = self.norm(x_out[:, 0, :])
        return fused

fuser = StaticImageFusion(
    dim_clin=X_clin.shape[1],
    dim_img=1280,
    fused_dim=1024,
    hidden_dim=1024,
    nhead=8,
    num_layers=1
).to("cuda")
fuser.eval()

with torch.no_grad():
    processed_count = 0
    for i, pid in enumerate(ids):
        img_path = os.path.join(FUSED_EMBED_DIR, f"{pid}-Fused_embed.pt")
        if not os.path.isfile(img_path):
            continue

        clin_feat = X_clin[i].unsqueeze(0).unsqueeze(1).to("cuda")
        img_feat = torch.load(img_path).unsqueeze(1).to("cuda")
        fused = fuser(clin_feat, img_feat)
        out_path = os.path.join(OUTPUT_DIR, f"{pid}-StaticMultimodal.pt")
        torch.save(fused.cpu(), out_path)
        processed_count += 1

print(f"\nStatic multimodal feature fusion completed for {processed_count} patients. Saved to: {OUTPUT_DIR}")

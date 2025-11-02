import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

FEATURES_DIR = Path(r"path_to_HE_features")
OCCLUDE_BASE = Path(r"path_to_occlusion_slide_embed")
MODEL_DIR = Path(r"path_to_HE_disease_models")
OUTPUT_BASE = Path(r"path_to_output")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

DISEASES = ['T2DM.Insulin.Dependent', 'Cirrhosis', 'CAD', 'Stroke', 'Heart.Failure',
            'Arrhythmias', 'Death', 'PAD', 'CKD', 'Extrahepatic.tumors',
            'Hypothyroidism', 'PCOS', 'HCC']
DISEASE_TO_PLOT = 'the_disease_you_want_to_analysis'

device = torch.device('cuda')

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 32),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x_flat = x.view(x.size(0), -1)
        out = self.net(x_flat)
        return out.squeeze(-1)

model_path = MODEL_DIR / f"{DISEASE_TO_PLOT}.pth"
if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = MLPClassifier(768).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
sigmoid = torch.nn.Sigmoid()

slide_dirs = sorted([d for d in OCCLUDE_BASE.iterdir() if d.is_dir()])
for slide_dir in slide_dirs:
    slide_id = slide_dir.name

    orig_fp = FEATURES_DIR / f"{slide_id}_embed.pt"
    if not orig_fp.exists():
        continue
    try:
        orig_feat = torch.load(orig_fp).float().to(device)
    except:
        continue
    if orig_feat.dim() == 1:
        orig_feat = orig_feat.unsqueeze(0)
    with torch.no_grad():
        orig_logit = model(orig_feat)
        orig_prob = float(sigmoid(orig_logit).cpu().item())

    meta_csv = slide_dir / f"{slide_id}_occlusion_metadata.csv"
    if not meta_csv.exists():
        continue
    try:
        meta = pd.read_csv(meta_csv)
    except:
        continue

    rows_out = []
    missing_count = 0
    for row in tqdm(meta.itertuples(index=False), desc=slide_id, leave=False):
        try:
            tile_index = int(row.tile_index)
            x = int(row.x)
            y = int(row.y)
            fname = str(row.filename)
        except:
            vals = list(row)
            if len(vals) >= 4:
                tile_index, x, y, fname = int(vals[0]), int(vals[1]), int(vals[2]), str(vals[3])
            else:
                continue

        feat_path = slide_dir / fname
        if not feat_path.exists():
            missing_count += 1
            rows_out.append({'tile_index': tile_index,
                             'x': x, 'y': y, 'filename': fname,
                             'orig_prob': orig_prob,
                             'occl_prob': np.nan,
                             'delta_prob': np.nan})
            continue

        try:
            occl_feat = torch.load(feat_path).float().to(device)
        except:
            rows_out.append({'tile_index': tile_index,
                             'x': x, 'y': y, 'filename': fname,
                             'orig_prob': orig_prob,
                             'occl_prob': np.nan,
                             'delta_prob': np.nan})
            continue

        if occl_feat.dim() == 1:
            occl_feat = occl_feat.unsqueeze(0)
        with torch.no_grad():
            occl_logit = model(occl_feat)
            occl_prob = float(sigmoid(occl_logit).cpu().item())

        delta_prob = float(orig_prob - occl_prob)
        rows_out.append({'tile_index': tile_index,
                         'x': x, 'y': y, 'filename': fname,
                         'orig_prob': orig_prob,
                         'occl_prob': occl_prob,
                         'delta_prob': delta_prob})

    out_dir = OUTPUT_BASE / slide_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{slide_id}_prob_deltas_{DISEASE_TO_PLOT}.csv"
    df_out = pd.DataFrame(rows_out,
                          columns=['tile_index', 'x', 'y', 'filename', 'orig_prob', 'occl_prob', 'delta_prob'])
    df_out.to_csv(out_csv, index=False)

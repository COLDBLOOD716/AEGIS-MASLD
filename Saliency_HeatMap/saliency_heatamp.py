import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import openslide
import cv2
from matplotlib import colormaps
from pathlib import Path
from tqdm import tqdm
from PIL import Image

FEATURES_DIR = Path(r"path_to_wsi_features")
OCCLUDE_BASE = Path(r"path_to_occlusion_slide_embed")
MODEL_DIR = Path(r"path_to_wsi_disease_models")
SLIDE_SVS = Path(r"path_to_svs")
OUTPUT_BASE = Path(r"path_to_output")
OUTPUT_BASE.mkdir(exist_ok=True)

DISEASES = ['T2DM','Cirrhosis','CAD','Stroke','Heart Failure',
            'Arrhythmias','Dyslipidemia','PAD','CKD','Hypertension',
            'Hypothyroidism','PCOS','HCC']
DISEASE_TO_PLOT = 'the_disease_you_want_to_plot'
DISPLAY_SCALE = 0.4
ALPHA = 0.5
TILE_SIZE = 256
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
        return self.net(x.view(x.size(0), -1)).squeeze(-1)

models = {}
for ds in DISEASES:
    mp = MODEL_DIR / f"{ds}.pth"
    m = MLPClassifier(768).to(device)
    m.load_state_dict(torch.load(mp, map_location=device))
    m.eval()
    models[ds] = m

slide_ids = sorted([d.name for d in OCCLUDE_BASE.iterdir() if d.is_dir()])
for slide_id in slide_ids:
    orig_fp = FEATURES_DIR / f"{slide_id}_embed.pt"
    if not orig_fp.exists():
        continue
    orig_feat = torch.load(orig_fp).float().to(device)
    with torch.no_grad():
        orig_probs = {ds: torch.sigmoid(models[ds](orig_feat)).item() for ds in DISEASES}

    meta_csv = OCCLUDE_BASE / slide_id / f"{slide_id}_occlusion_metadata.csv"
    if not meta_csv.exists():
        continue
    meta = pd.read_csv(meta_csv)

    records = []
    for row in tqdm(meta.itertuples(), desc=slide_id, leave=False):
        idx, x, y, fname = row.tile_index, row.x, row.y, row.filename
        feat = torch.load(OCCLUDE_BASE / slide_id / fname).float().to(device)
        with torch.no_grad():
            prob = torch.sigmoid(models[DISEASE_TO_PLOT](feat)).item()
        delta = abs(orig_probs[DISEASE_TO_PLOT] - prob)
        records.append({'tile_index':idx, 'x':x, 'y':y, 'delta':delta})

    out_dir = OUTPUT_BASE / slide_id
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(out_dir / f"{slide_id}_heatmap_scores.csv", index=False)

    svs_path = SLIDE_SVS / f"{slide_id}.svs"
    try:
        slide = openslide.OpenSlide(str(svs_path))
    except:
        continue

    target_ds = 1 / DISPLAY_SCALE
    level = slide.get_best_level_for_downsample(target_ds)
    ds = slide.level_downsamples[level]
    disp_W, disp_H = slide.level_dimensions[level]

    base_img = slide.read_region((0,0), level, slide.level_dimensions[level]).convert("RGB")
    base = np.array(base_img, dtype=np.uint8)
    slide.close()

    heat_rgba = np.zeros((disp_H, disp_W, 4), dtype=np.uint8)
    scores = df['delta'].values
    vmin, vmax = np.percentile(scores, [5,95])
    norm = np.clip((scores - vmin)/(vmax - vmin), 0, 1)
    cols = (colormaps.get_cmap('jet')(norm)[:,:3] * 255).astype(np.uint8)

    w0 = int((TILE_SIZE / ds) * 2)
    h0 = int((TILE_SIZE / ds) * 2)
    threshold = 0.75

    for (x, y), nrm, col in zip(df[['x', 'y']].values, norm, cols):
        x0 = int(x / ds) - (w0 // 2 - int(TILE_SIZE / (2 * ds)))
        y0 = int(y / ds) - (h0 // 2 - int(TILE_SIZE / (2 * ds)))
        x0, y0 = max(0, x0), max(0, y0)

        if nrm > threshold:
            patch_rgb = [0, 0, 0]
            patch_alpha = 255
        else:
            patch_rgb = col.tolist()
            patch_alpha = int(ALPHA * 255)

        heat_rgba[y0:y0 + h0, x0:x0 + w0, :3] = patch_rgb
        heat_rgba[y0:y0 + h0, x0:x0 + w0, 3] = patch_alpha

    out_pdf = out_dir / f"{slide_id}_occlusion_{DISEASE_TO_PLOT}.pdf"
    rgba_img = cv2.cvtColor(heat_rgba, cv2.COLOR_RGBA2BGRA)
    temp_png = out_pdf.with_suffix('.png')
    success = cv2.imwrite(str(temp_png), rgba_img)

    if success:
        img = Image.open(temp_png)
        img.save(out_pdf)
        os.remove(temp_png)

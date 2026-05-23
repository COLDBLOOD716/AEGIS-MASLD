# AEGIS-MASLD

AEGIS-MASLD is an open-source workflow for MASLD whole-slide image analysis and longitudinal outcome prediction. The repository contains three major modules:

- **Multi-outcome prediction**: dual-stain WSI embeddings, clinical variables, and longitudinal lifestyle records are fused and passed to Trans-LSTM models for 15-year disease-risk prediction.
- **Tissue phenotype clustering and visualization**: tile-level embeddings are clustered to generate global UMAP plots and per-slide spatial phenotype maps.
- **Cell segmentation and classification**: Hover-Net is used to segment and classify nuclei/cells on whole-slide images.

This README gives a step-by-step two-case demo for reviewers and readers. The demo uses the four test slides in `Test_WSIs/`:

- `TEST01-HE.svs`
- `TEST01-SR.svs`
- `TEST02-HE.svs`
- `TEST02-SR.svs`

The expected patient IDs are `TEST01` and `TEST02`.

## 1. Environment Setup

AEGIS uses PyTorch, OpenSlide, GigaPath-related WSI utilities, Scanpy, scikit-learn, and Hover-Net dependencies. A CUDA-enabled GPU is recommended for the WSI and Trans-LSTM steps.

```bash
# Create and activate a clean environment.
conda create -n aegis-masld python=3.11 -y
conda activate aegis-masld

# Install the Python dependencies listed by this repository.
pip install -r requirements.txt

# OpenSlide is required for reading .svs whole-slide images.
# Linux example:
sudo apt-get update
sudo apt-get install -y openslide-tools libopenslide-dev

# macOS example:
brew install openslide
```

GigaPath model loading requires a Hugging Face access token. Set it before running WSI feature extraction:

```bash
export HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"
```

## 2. Prepare the Two-Case Demo Workspace

Run the following Python snippet from the repository root. It creates a reproducible demo folder, links the four test slides, and defines where each downstream output should be written.

```python
from pathlib import Path
import os
import shutil

# Repository root. Run this script from the root of AEGIS-MASLD.
repo = Path.cwd()

# All demo outputs will be kept under demo_runs/two_case_demo.
demo = repo / "demo_runs" / "two_case_demo"
slide_dir = demo / "slides"
dyn_dir = demo / "dynamic_lifestyle"
clin_dir = demo / "clinical"
he_embed_dir = demo / "embeddings" / "HE"
sr_embed_dir = demo / "embeddings" / "SR"
tile_dir = demo / "tile_features"
dual_fused_dir = demo / "fused_dual_stain"
static_multimodal_dir = demo / "static_multimodal"
prediction_dir = demo / "predictions"
cluster_dir = demo / "tissue_clusters"
occlusion_dir = demo / "occlusion_embeddings"
saliency_dir = demo / "saliency_heatmaps"
cell_dir = demo / "cell_segmentation"

for path in [
    slide_dir, dyn_dir, clin_dir, he_embed_dir, sr_embed_dir, tile_dir,
    dual_fused_dir, static_multimodal_dir, prediction_dir, cluster_dir,
    occlusion_dir, saliency_dir, cell_dir
]:
    path.mkdir(parents=True, exist_ok=True)

# Link or copy the provided test slides into the demo folder.
for name in [
    "TEST01-HE.svs",
    "TEST01-SR.svs",
    "TEST02-HE.svs",
    "TEST02-SR.svs",
]:
    src = repo / "Test_WSIs" / name
    dst = slide_dir / name
    if dst.exists():
        continue
    try:
        os.symlink(src, dst)
    except OSError:
        # Fall back to copying on systems where symlinks are unavailable.
        shutil.copy2(src, dst)

print(f"Demo workspace: {demo}")
```

Place the longitudinal lifestyle CSV files in:

```text
demo_runs/two_case_demo/dynamic_lifestyle/TEST01.csv
demo_runs/two_case_demo/dynamic_lifestyle/TEST02.csv
```

Place the two-case clinical table in:

```text
demo_runs/two_case_demo/clinical/test_clinical_records.csv
```

The clinical table must contain one row per patient and an `ID` column. `Feature_fusion/Clinical_fuse.py` expects these columns:

```text
ID, HBsAg, HBsAb, HBeAg, HBeAb, HBcAb,
ALT, AST, GGT, TBIL, DBIL, Creatinine,
Uric Acid, Blood glucose, TG, TC, HDL, LDL
```

The longitudinal lifestyle files are read with `pd.read_csv(path, index_col=0)`. The first 10 columns are used as dynamic predictors, and outcome columns are used only when labels are available.

## 3. Extract WSI Embeddings with GigaPath

Edit the path variables near the top of `wsi_inference/run_gigapath.py`:

```python
# Input folder containing the four .svs files.
source_dir = r"demo_runs/two_case_demo/slides"

# Slide-level embeddings are written here.
slide_embed_output_dir = r"demo_runs/two_case_demo/embeddings/all_slides"

# Temporary tile images, tile embeddings, and tile coordinates are written here.
tmp_dir_base = r"demo_runs/two_case_demo/tile_features"
```

Then run:

```bash
python wsi_inference/run_gigapath.py
```

Expected outputs include:

```text
demo_runs/two_case_demo/embeddings/all_slides/TEST01-HE_embed.pt
demo_runs/two_case_demo/embeddings/all_slides/TEST01-SR_embed.pt
demo_runs/two_case_demo/embeddings/all_slides/TEST02-HE_embed.pt
demo_runs/two_case_demo/embeddings/all_slides/TEST02-SR_embed.pt
demo_runs/two_case_demo/embeddings/all_slides/inference_timings.csv
demo_runs/two_case_demo/tile_features/<slide_id>/<slide_id>_tile_coords.csv
```

Separate the HE and SR embeddings before dual-stain fusion:

```python
from pathlib import Path
import shutil

demo = Path("demo_runs/two_case_demo")
all_embed_dir = demo / "embeddings" / "all_slides"
he_dir = demo / "embeddings" / "HE"
sr_dir = demo / "embeddings" / "SR"
he_dir.mkdir(parents=True, exist_ok=True)
sr_dir.mkdir(parents=True, exist_ok=True)

for pt in all_embed_dir.glob("*_embed.pt"):
    if pt.name.endswith("-HE_embed.pt"):
        shutil.copy2(pt, he_dir / pt.name)
    elif pt.name.endswith("-SR_embed.pt"):
        shutil.copy2(pt, sr_dir / pt.name)
```

## 4. Fuse HE and SR WSI Embeddings

Edit `Feature_fusion/Dual_stain_fuse.py`:

```python
class Config:
    he_dir = r"demo_runs/two_case_demo/embeddings/HE"
    sr_dir = r"demo_runs/two_case_demo/embeddings/SR"
    output_dir = r"demo_runs/two_case_demo/fused_dual_stain"
    fused_dim = 1280
    device = "cuda"
```

Run:

```bash
python Feature_fusion/Dual_stain_fuse.py
```

Expected outputs:

```text
demo_runs/two_case_demo/fused_dual_stain/TEST01-Fused_embed.pt
demo_runs/two_case_demo/fused_dual_stain/TEST02-Fused_embed.pt
demo_runs/two_case_demo/fused_dual_stain/contribution_records.csv
```

## 5. Fuse Clinical Variables with WSI Embeddings

Edit `Feature_fusion/Clinical_fuse.py`:

```python
CLINICAL_CSV = r"demo_runs/two_case_demo/clinical/test_clinical_records.csv"
FUSED_EMBED_DIR = r"demo_runs/two_case_demo/fused_dual_stain"
OUTPUT_DIR = r"demo_runs/two_case_demo/static_multimodal"
```

Run:

```bash
python Feature_fusion/Clinical_fuse.py
```

Expected outputs:

```text
demo_runs/two_case_demo/static_multimodal/TEST01-StaticMultimodal.pt
demo_runs/two_case_demo/static_multimodal/TEST02-StaticMultimodal.pt
```

## 6. Run Multi-Outcome Prediction

Edit the path variables in `Trans-LSTM_model/predict_with_Trans-LSTM.py`:

```python
DEVICE = torch.device("cuda")

NEW_STATIC_DIR = r"demo_runs/two_case_demo/static_multimodal"
DYN_DIR        = r"demo_runs/two_case_demo/dynamic_lifestyle"
MODEL_DIR      = r"Trans-LSTM_model"
OUTPUT_DIR     = r"demo_runs/two_case_demo/predictions"

# Optional. If no external labels are available, keep this as a non-existing path.
EXTERNAL_LABELS_CSV = r"demo_runs/two_case_demo/clinical/test_clinical_records.csv"
```

Run:

```bash
python Trans-LSTM_model/predict_with_Trans-LSTM.py
```

Expected prediction outputs:

```text
demo_runs/two_case_demo/predictions/TEST01_pred.csv
demo_runs/two_case_demo/predictions/TEST02_pred.csv
demo_runs/two_case_demo/predictions/pred_15yr_summary.csv
demo_runs/two_case_demo/predictions/summary_15yr_prevalence.csv
```

Each `<ID>_pred.csv` contains 15 yearly probabilities for the diseases listed in the script. `pred_15yr_summary.csv` contains each disease's 15-year binary prediction and maximum predicted probability for each case.

## 7. Run Tissue Phenotype Clustering

Edit `wsi_tile_cluster/run_Barlowtwins.py`:

```python
base_dir = Path(r"demo_runs/two_case_demo/tile_features")
coords_base = Path(r"demo_runs/two_case_demo/tile_features")
output_dir = Path(r"demo_runs/two_case_demo/tissue_clusters")
```

Run:

```bash
python wsi_tile_cluster/run_Barlowtwins.py
```

Expected visualization outputs:

```text
demo_runs/two_case_demo/tissue_clusters/global_tiles_umap_labeled.pdf
demo_runs/two_case_demo/tissue_clusters/TEST01-HE_spatial_clusters_labeled.pdf
demo_runs/two_case_demo/tissue_clusters/TEST02-HE_spatial_clusters_labeled.pdf
```

[The Tissue Phenotype Clustering Results of TEST01-HE](https://github.com/COLDBLOOD716/AEGIS-MASLD/blob/main/images/TEST01_Cluster.png?raw=true)

[The Tissue Phenotype Clustering Results of TEST02-HE](https://github.com/COLDBLOOD716/AEGIS-MASLD/blob/main/images/TEST02_Cluster.png?raw=true)

## 8. Generate Saliency Heatmaps by Tile Occlusion

First, create leave-one-tile-out slide embeddings. Edit `Saliency_HeatMap/run_tile_occlussion.py`:

```python
tile_feature_base = r"demo_runs/two_case_demo/tile_features"
slide_embed_output_base = r"demo_runs/two_case_demo/occlusion_embeddings"
```

Run:

```bash
python Saliency_HeatMap/run_tile_occlussion.py
```

Second, compute per-tile probability gaps. Edit `Saliency_HeatMap/Cal_occlusion_probgap.py`:

```python
FEATURES_DIR = Path(r"demo_runs/two_case_demo/embeddings/all_slides")
OCCLUDE_BASE = Path(r"demo_runs/two_case_demo/occlusion_embeddings")
# This folder must contain WSI-level MLP disease classifiers whose state_dict
# matches the MLPClassifier(input_dim=768) defined in this script.
# The Trans-LSTM_model/ folder is used for longitudinal prediction and is not
# shape-compatible with this saliency script.
MODEL_DIR = Path(r"path_to_wsi_mlp_disease_models")
OUTPUT_BASE = Path(r"demo_runs/two_case_demo/saliency_heatmaps")

# Replace this with one supported disease model, for example:
DISEASE_TO_PLOT = "HCC"
```

Run:

```bash
python Saliency_HeatMap/Cal_occlusion_probgap.py
```

Third, render heatmaps on the WSI canvas. Edit `Saliency_HeatMap/saliency_heatamp.py`:

```python
FEATURES_DIR = Path(r"demo_runs/two_case_demo/embeddings/all_slides")
OCCLUDE_BASE = Path(r"demo_runs/two_case_demo/occlusion_embeddings")
# Use the same WSI-level MLP disease-model directory as above.
MODEL_DIR = Path(r"path_to_wsi_mlp_disease_models")
SLIDE_SVS = Path(r"demo_runs/two_case_demo/slides")
OUTPUT_BASE = Path(r"demo_runs/two_case_demo/saliency_heatmaps")

# Use the same disease as in Cal_occlusion_probgap.py.
DISEASE_TO_PLOT = "HCC"
```

Run:

```bash
python Saliency_HeatMap/saliency_heatamp.py
```

Expected saliency outputs include per-tile probability-gap CSV files and disease-specific PDF heatmaps:

```text
demo_runs/two_case_demo/saliency_heatmaps/<slide_id>/<slide_id>_prob_deltas_HCC.csv
demo_runs/two_case_demo/saliency_heatmaps/<slide_id>/<slide_id>_occlusion_HCC.pdf
```

## 9. Run Cell Segmentation and Classification

The cell module uses Hover-Net. Make sure the `hover_net` dependency folder, `type_info.json`, and pretrained weights are available under `Cell_segement_and_classify/`.

Example command:

```bash
cd Cell_segement_and_classify/hover_net

python run_infer.py \
  --gpu=0 \
  --nr_types=6 \
  --type_info_path="type_info.json" \
  --model_path="hovernet_fast_pannuke_type_tf2pytorch.tar" \
  wsi \
  --input_dir="../../Test_WSIs" \
  --output_dir="../../demo_runs/two_case_demo/cell_segmentation" \
  --cache_path="../../demo_runs/two_case_demo/cell_segmentation/cache"
```

Typical outputs include cell instance maps, classified nuclei/cell annotations, and WSI-level inference artifacts under:

```text
demo_runs/two_case_demo/cell_segmentation
```

## 10. Minimal Output Checklist

After completing the demo, reviewers should be able to inspect:

```text
demo_runs/two_case_demo/predictions/TEST01_pred.csv
demo_runs/two_case_demo/predictions/TEST02_pred.csv
demo_runs/two_case_demo/predictions/pred_15yr_summary.csv
demo_runs/two_case_demo/tissue_clusters/global_tiles_umap_labeled.pdf
demo_runs/two_case_demo/tissue_clusters/*_spatial_clusters_labeled.pdf
demo_runs/two_case_demo/saliency_heatmaps/*/*_occlusion_*.pdf
demo_runs/two_case_demo/cell_segmentation/
```

## Notes for Reproducibility

- Keep patient IDs consistent across WSI filenames, clinical rows, dynamic lifestyle CSV filenames, and output feature names.
- The demo scripts currently use explicit path variables at the top of each file. Update those variables before running each step.
- The saliency scripts expect WSI-level MLP classifier weights. They are separate from the Trans-LSTM longitudinal outcome models.
- If CUDA is unavailable, change `torch.device("cuda")` or `device = "cuda"` to CPU, but WSI embedding and saliency steps will be much slower.
- For large WSI files, keep tile and occlusion outputs on a drive with sufficient free space.
- Set `HF_TOKEN` in the shell instead of hard-coding a private token in source code.

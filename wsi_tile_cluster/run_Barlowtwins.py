import os
import torch
import numpy as np
import scanpy as sc
import anndata
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

base_dir = Path(r'PATH_TO_TILE_FEATURES')
coords_base = Path(r'PATH_TO_TILE_COORDS')
output_dir = Path(r'PATH_TO_OUTPUT')
output_dir.mkdir(parents=True, exist_ok=True)

embeds_list = []
coords_list = []
slide_ids = []

pt_paths = sorted(base_dir.glob("*/" + "*_tile_outputs.pt"))

for pt_path in tqdm(pt_paths, desc="Loading Barlow tile features"):
    slide_id = pt_path.stem.replace("_tile_outputs", "")
    emb_tensor = torch.load(pt_path)
    emb = emb_tensor.cpu().numpy()
    n_tiles = emb.shape[0]

    coords_csv = coords_base / slide_id / f"{slide_id}_tile_coords.csv"
    if not coords_csv.exists():
        continue

    crd = np.loadtxt(coords_csv, delimiter=",", skiprows=1, dtype=int)
    if crd.shape[0] != n_tiles:
        raise ValueError(f"Tile count mismatch for {slide_id}")

    embeds_list.append(emb)
    coords_list.append(crd)
    slide_ids += [slide_id] * n_tiles

all_embeds = np.vstack(embeds_list)
all_coords = np.vstack(coords_list)
slide_ids = np.array(slide_ids)

adata = anndata.AnnData(X=all_embeds)
adata.obs['slide'] = slide_ids.astype(str)

sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')
sc.tl.leiden(adata, resolution=0.6)
sc.tl.umap(adata)

clusters = adata.obs['leiden'].astype(int).values
np.save(output_dir / "all_tile_coords.npy", all_coords)
np.save(output_dir / "all_tile_clusters.npy", clusters)

n_clusters = len(np.unique(clusters))

plt.figure(figsize=(8, 6))
ax = sc.pl.umap(
    adata,
    color='leiden',
    size=0.2,
    show=False,
    title=f"Global UMAP ({n_clusters} clusters)",
    legend_loc='right margin',
    legend_fontsize=6,
    return_fig=True
).axes[0]

umap_coords = adata.obsm['X_umap']
leiden_clusters = adata.obs['leiden'].astype(int).values

for cid in range(n_clusters):
    idx = (leiden_clusters == cid)
    if np.sum(idx) == 0:
        continue
    x_mean = np.median(umap_coords[idx, 0])
    y_mean = np.median(umap_coords[idx, 1])
    ax.text(
        x_mean, y_mean, str(cid),
        fontsize=7, weight='bold', color='black',
        ha='center', va='center',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, boxstyle='round,pad=0.2')
    )

plt.tight_layout()
plt.savefig(output_dir / "global_tiles_umap_labeled.pdf", dpi=300, format='pdf')
plt.close()

unique_slides = np.unique(slide_ids)

for slide in tqdm(unique_slides, desc="Drawing per-slide spatial cluster plots"):
    mask = (adata.obs['slide'] == slide).values
    coords = all_coords[mask]
    clusters_slide = clusters[mask]

    np.save(output_dir / f"{slide}_coords.npy", coords)
    np.save(output_dir / f"{slide}_clusters.npy", clusters_slide)

    xs = np.unique(coords[:, 0])
    tile_dx = np.median(np.diff(xs)) if len(xs) > 1 else 256
    data_range_x = coords[:, 0].ptp() if coords[:, 0].ptp() > 0 else 1
    fig_w_pts = 6 * 72
    s_pts = (tile_dx / data_range_x * fig_w_pts) ** 2

    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=clusters_slide,
        s=s_pts,
        marker='s',
        alpha=1.0,
        cmap='tab20'
    )

    for cid in np.unique(clusters_slide):
        idx = (clusters_slide == cid)
        x_c = np.median(coords[idx, 0])
        y_c = np.median(coords[idx, 1])
        ax.text(
            x_c, y_c, str(cid),
            fontsize=7, weight='bold', color='black',
            ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, boxstyle='round,pad=0.2')
        )

    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title(f"Slide {slide} spatial clusters")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(
        *scatter.legend_elements(),
        loc='center left',
        bbox_to_anchor=(1.0, 0.5),
        title="Cluster ID",
        fontsize=6
    )
    plt.tight_layout()
    plt.savefig(output_dir / f"{slide}_spatial_clusters_labeled.pdf", dpi=300, format='pdf')
    plt.close(fig)

np.save(output_dir / "all_tile_umap_coords.npy", adata.obsm['X_umap'])
np.save(output_dir / "all_tile_slide_ids.npy", adata.obs['slide'].values)

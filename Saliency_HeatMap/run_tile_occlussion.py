import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["HF_TOKEN"] = "your_hf_token_here"
assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable"

import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from gigapath.pipeline import load_tile_slide_encoder, run_inference_with_slide_encoder
import pandas as pd

tile_feature_base = r"path_to_tile_features"
slide_embed_output_base = r"path_to_slide_embeddings"
os.makedirs(slide_embed_output_base, exist_ok=True)

tile_encoder, slide_encoder_model = load_tile_slide_encoder(global_pool=True)

slide_ids = [p.name for p in Path(tile_feature_base).iterdir() if p.is_dir()]

for slide_id in tqdm(slide_ids, desc="Processing slides"):
    try:
        pt_path = Path(tile_feature_base) / slide_id / f"{slide_id}_tile_outputs.pt"
        coords_path = Path(tile_feature_base) / slide_id / f"{slide_id}_tile_coords.csv"

        if not pt_path.exists() or not coords_path.exists():
            continue

        tile_data = torch.load(pt_path)
        tile_embeds = tile_data['tile_embeds']
        coords = tile_data['coords']

        if 'attention_mask' in tile_data:
            attention_mask = tile_data['attention_mask']
        else:
            attention_mask = torch.ones(tile_embeds.shape[0], dtype=torch.bool)

        num_tiles = tile_embeds.shape[0]
        out_dir = Path(slide_embed_output_base) / slide_id
        out_dir.mkdir(parents=True, exist_ok=True)

        record_list = []

        for i in range(num_tiles):
            keep_mask = torch.ones(num_tiles, dtype=torch.bool)
            keep_mask[i] = False

            masked_inputs = {
                'tile_embeds': tile_embeds[keep_mask],
                'coords': coords[keep_mask],
            }

            slide_embeds = run_inference_with_slide_encoder(
                slide_encoder_model=slide_encoder_model,
                **masked_inputs
            )
            last_layer_embed = slide_embeds['last_layer_embed']

            output_path = out_dir / f"{slide_id}_occlude_{i}.pt"
            torch.save(last_layer_embed, output_path)

            x, y = coords[i].tolist()
            record_list.append({'tile_index': i, 'x': int(x), 'y': int(y), 'filename': output_path.name})

        df = pd.DataFrame(record_list)
        df.to_csv(out_dir / f"{slide_id}_occlusion_metadata.csv", index=False)

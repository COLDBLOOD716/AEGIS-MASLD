import os
import torch
import pandas as pd
import openslide
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime
from gigapath.preprocessing.data.slide_utils import find_level_for_target_mpp
from gigapath.pipeline import (
    tile_one_slide,
    load_tile_slide_encoder,
    run_inference_with_tile_encoder,
    run_inference_with_slide_encoder,
)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN"
assert "HF_TOKEN" in os.environ

def get_level_for_mpp(slide_path, target_mpp=0.5):
    try:
        lvl = find_level_for_target_mpp(slide_path, target_mpp)
        if lvl is not None:
            return lvl
    except Exception:
        pass
    slide = openslide.OpenSlide(slide_path)
    mpp_x = float(slide.properties.get('openslide.mpp-x', 0))
    slide.close()
    if mpp_x <= 0:
        raise ValueError(f"No valid MPP in slide properties: {slide_path}")
    slide = openslide.OpenSlide(slide_path)
    downsamples = slide.level_downsamples
    slide.close()
    diffs = [abs(mpp_x * d - target_mpp) for d in downsamples]
    return int(np.argmin(diffs))

source_dir = r'PATH_TO_SOURCE_SLIDES'
slide_embed_output_dir = r'PATH_TO_SLIDE_EMBED_OUTPUT'
tmp_dir_base = r'PATH_TO_TEMP_TILE_DIR'
os.makedirs(slide_embed_output_dir, exist_ok=True)

tile_encoder, slide_encoder_model = load_tile_slide_encoder(global_pool=True)

svs_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.svs')]
records = []
overall_start_time = None
overall_end_time = None

for svs_file in tqdm(svs_files, desc="Processing slides"):
    slide_path = os.path.join(source_dir, svs_file)
    slide_id   = os.path.splitext(svs_file)[0]
    tmp_dir    = os.path.join(tmp_dir_base, slide_id)
    os.makedirs(tmp_dir, exist_ok=True)

    target_mpp = 0.5
    level = get_level_for_mpp(slide_path, target_mpp)
    if level is None:
        records.append({
            "slide_id": slide_id,
            "start_time": "",
            "end_time": "",
            "duration_sec": 0.0,
            "status": "skipped_no_level",
            "error": ""
        })
        continue

    if overall_start_time is None:
        overall_start_time = time.time()

    slide_start = time.time()
    slide_start_iso = datetime.fromtimestamp(slide_start).isoformat(timespec='seconds')
    status = "success"
    error_msg = ""

    try:
        tile_one_slide(slide_path, save_dir=tmp_dir, level=level)
        slide_dir = os.path.join(tmp_dir, "output", os.path.basename(slide_path))
        image_paths = [
            os.path.join(slide_dir, img)
            for img in os.listdir(slide_dir)
            if img.lower().endswith('.png')
        ]
        if not image_paths:
            raise RuntimeError("No tiles found after tiling")

        tile_encoder_outputs = run_inference_with_tile_encoder(image_paths, tile_encoder)
        torch.save(
            tile_encoder_outputs,
            os.path.join(tmp_dir, f"{slide_id}_tile_outputs.pt")
        )

        coords_array = tile_encoder_outputs['coords'].cpu().numpy()
        np.savetxt(
            os.path.join(tmp_dir, f"{slide_id}_tile_coords.csv"),
            coords_array,
            delimiter=",",
            header="x,y",
            comments=""
        )

        slide_embeds = run_inference_with_slide_encoder(
            slide_encoder_model=slide_encoder_model,
            **tile_encoder_outputs
        )
        last_layer_embed = slide_embeds['last_layer_embed']
        torch.save(
            last_layer_embed,
            os.path.join(slide_embed_output_dir, f"{slide_id}_embed.pt")
        )

    except Exception as e:
        status = "error"
        error_msg = repr(e)[:300]

    finally:
        slide_end = time.time()
        slide_end_iso = datetime.fromtimestamp(slide_end).isoformat(timespec='seconds')
        duration = slide_end - slide_start
        records.append({
            "slide_id": slide_id,
            "start_time": slide_start_iso,
            "end_time": slide_end_iso,
            "duration_sec": float(f"{duration:.4f}"),
            "status": status,
            "error": error_msg
        })
        overall_end_time = slide_end

if overall_start_time is None:
    overall_record = {
        "slide_id": "__OVERALL__",
        "start_time": "",
        "end_time": "",
        "duration_sec": 0.0,
        "status": "no_slides_processed",
        "error": ""
    }
else:
    overall_duration = overall_end_time - overall_start_time
    overall_record = {
        "slide_id": "__OVERALL__",
        "start_time": datetime.fromtimestamp(overall_start_time).isoformat(timespec='seconds'),
        "end_time": datetime.fromtimestamp(overall_end_time).isoformat(timespec='seconds'),
        "duration_sec": float(f"{overall_duration:.4f}"),
        "status": "completed",
        "error": ""
    }

records.append(overall_record)
df = pd.DataFrame.from_records(records, columns=[
    "slide_id", "start_time", "end_time", "duration_sec", "status", "error"
])
out_csv = os.path.join(slide_embed_output_dir, "inference_timings.csv")
df.to_csv(out_csv, index=False)
print(f"Timing results saved to: {out_csv}")

# ------------------------------------------------------------------------------------------
# create_tiles_dataset.py  （修改版）
# ------------------------------------------------------------------------------------------
import os
# 强制 matplotlib 使用非交互后端（必须在导入 pyplot 之前）
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib
matplotlib.use('Agg')

import functools
import logging
import shutil
import tempfile
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import PIL
from PIL import Image, ImageDraw
# 尽量避免在后台线程使用 matplotlib.pyplot；我们使用 PIL 做可视化
from monai.data import Dataset
from monai.data.wsi_reader import WSIReader
from openslide import OpenSlide
from tqdm import tqdm

from gigapath.preprocessing.data import tiling
from gigapath.preprocessing.data.foreground_segmentation import LoadROId, segment_foreground

# ---------------------------
# 保持原有函数（未改动逻辑）
# ---------------------------
def select_tiles(foreground_mask: np.ndarray, occupancy_threshold: float) \
        -> Tuple[np.ndarray, np.ndarray]:
    if occupancy_threshold < 0. or occupancy_threshold > 1.:
        raise ValueError("Tile occupancy threshold must be between 0 and 1")
    occupancy = foreground_mask.mean(axis=(-2, -1), dtype=np.float16)
    return (occupancy > occupancy_threshold).squeeze(), occupancy.squeeze()  # type: ignore


def get_tile_descriptor(tile_location: Sequence[int]) -> str:
    return f"{tile_location[0]:05d}x_{tile_location[1]:05d}y"


def get_tile_id(slide_id: str, tile_location: Sequence[int]) -> str:
    return f"{slide_id}.{get_tile_descriptor(tile_location)}"


def save_image(array_chw: np.ndarray, path: Path) -> PIL.Image.Image:
    """Save an image array in (C, H, W) format to disk. 返回 PIL Image 对象"""
    path.parent.mkdir(parents=True, exist_ok=True)
    array_hwc = np.moveaxis(array_chw, 0, -1).astype(np.uint8).squeeze()
    pil_image = PIL.Image.fromarray(array_hwc)
    pil_image = pil_image.convert('RGB')
    pil_image.save(path)
    return pil_image


def check_empty_tiles(tiles: np.ndarray, std_th: int = 5, extreme_value_portion_th: float = 0.5) -> np.ndarray:
    b, c, h, w = tiles.shape
    flattned_tiles = tiles.reshape(b, c, h * w)
    std_rgb = flattned_tiles[:, :, :].std(axis=2)
    std_rgb_mean = std_rgb.mean(axis=1)
    low_std_mask = std_rgb_mean < std_th
    extreme_value_count = ((flattned_tiles == 0)).sum(axis=2)
    extreme_value_proportion = extreme_value_count / (h * w)
    extreme_value_mask = extreme_value_proportion.max(axis=1) > extreme_value_portion_th
    return low_std_mask | extreme_value_mask


def generate_tiles(slide_image: np.ndarray, tile_size: int, foreground_threshold: float,
                   occupancy_threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    image_tiles, tile_locations = tiling.tile_array_2d(slide_image, tile_size=tile_size,
                                                       constant_values=255)
    logging.info(f"image_tiles.shape: {image_tiles.shape}, dtype: {image_tiles.dtype}")
    logging.info(f"Tiled {slide_image.shape} to {image_tiles.shape}")
    foreground_mask, _ = segment_foreground(image_tiles, foreground_threshold)
    selected, occupancies = select_tiles(foreground_mask, occupancy_threshold)
    n_discarded = (~selected).sum()
    logging.info(f"Percentage tiles discarded: {n_discarded / len(selected) * 100:.2f}")

    image_tiles = image_tiles[selected]
    tile_locations = tile_locations[selected]
    occupancies = occupancies[selected]

    if len(tile_locations) == 0:
        logging.warning("No tiles selected")
    else:
        logging.info(f"After filtering: min y: {tile_locations[:, 0].min()}, max y: {tile_locations[:, 0].max()}, min x: {tile_locations[:, 1].min()}, max x: {tile_locations[:, 1].max()}")

    return image_tiles, tile_locations, occupancies, n_discarded


def get_tile_info(sample: Dict["SlideKey", Any], occupancy: float, tile_location: Sequence[int],
                  rel_slide_dir: Path) -> Dict["TileKey", Any]:
    slide_id = sample["slide_id"]
    descriptor = get_tile_descriptor(tile_location)
    rel_image_path = f"{rel_slide_dir}/{descriptor}.png"

    tile_info = {
        "slide_id": slide_id,
        "tile_id": get_tile_id(slide_id, tile_location),
        "image": rel_image_path,
        "label": sample.get("label", None),
        "tile_x": tile_location[0],
        "tile_y": tile_location[1],
        "occupancy": occupancy,
        "metadata": {"slide_" + key: value for key, value in sample["metadata"].items()}
    }

    return tile_info


def format_csv_row(tile_info: Dict["TileKey", Any], keys_to_save: Iterable["TileKey"],
                   metadata_keys: Iterable[str]) -> str:
    tile_slide_metadata = tile_info.pop("metadata")
    fields = [str(tile_info[key]) for key in keys_to_save]
    fields.extend(str(tile_slide_metadata[key]) for key in metadata_keys)
    dataset_row = ','.join(fields)
    return dataset_row


def load_image_dict(sample: dict, level: int, margin: int, foreground_threshold: Optional[float] = None) -> Dict["SlideKey", Any]:
    loader = LoadROId(WSIReader(backend="OpenSlide"), level=level, margin=margin,
                      foreground_threshold=foreground_threshold)
    img = loader(sample)
    return img


def save_thumbnail(slide_path, output_path, size_target=1024):
    try:
        with OpenSlide(str(slide_path)) as openslide_obj:
            scale = size_target / max(openslide_obj.dimensions)
            thumbnail = openslide_obj.get_thumbnail([int(m * scale) for m in openslide_obj.dimensions])
            # 保存为 PNG
            thumbnail.save(output_path)
            logging.info(f"Saving thumbnail {output_path}, shape {thumbnail.size}")
    except Exception:
        logging.exception(f"Failed to save thumbnail for {slide_path} -> {output_path}")


# ----------------------------
# 使用 PIL 绘制 tile rectangle 的可视化函数（线程安全）
# ----------------------------
def visualize_tile_locations_pil(slide_sample, output_path, tile_info_list, tile_size, origin_offset):
    """
    用 PIL 在 ROI 缩略图上画出 tile rectangle。
    slide_sample["image"] expected shape: (C,H,W) numpy
    origin_offset: sample["origin"] (level-0 coordinate)
    """
    try:
        slide_image = slide_sample["image"]  # C,H,W
        downscale_factor = slide_sample["scale"]
        # 转换为 HWC uint8
        img_hwc = np.moveaxis(slide_image, 0, -1).astype(np.uint8)
        pil_img = Image.fromarray(img_hwc).convert("RGB")
        draw = ImageDraw.Draw(pil_img)

        for tile_info in tile_info_list:
            # tile_info tile_x/tile_y in level-0 coordinates
            xy = ((tile_info["tile_x"] - origin_offset[0]) / downscale_factor,
                  (tile_info["tile_y"] - origin_offset[1]) / downscale_factor)
            x0 = int(xy[0])
            y0 = int(xy[1])
            x1 = x0 + int(tile_size / downscale_factor) if downscale_factor != 0 else x0 + tile_size
            y1 = y0 + int(tile_size / downscale_factor) if downscale_factor != 0 else y0 + tile_size
            # draw rectangle (outline only)
            draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
        # 保存
        pil_img.save(output_path)
    except Exception:
        logging.exception(f"visualize_tile_locations_pil failed for {output_path}")


def is_already_processed(output_tiles_dir):
    if not output_tiles_dir.exists():
        return False
    if len(list(output_tiles_dir.glob("*.png"))) == 0:
        return False
    dataset_csv_path = output_tiles_dir / "dataset.csv"
    try:
        df = pd.read_csv(dataset_csv_path)
    except:
        return False
    return len(df) > 0


# ----------------------------
# process_slide：主要修改点
#  - 关闭 tile_progress（由调用处决定）
#  - 使用 PIL 保存 ROI 缩略图，避免 matplotlib 交互
#  - 使用 PIL 可视化 tile overlay，并在失败时捕获异常（不影响主流程）
# ----------------------------
def process_slide(sample: Dict["SlideKey", Any], level: int, margin: int, tile_size: int,
                  foreground_threshold: Optional[float], occupancy_threshold: float, output_dir: Path,
                  thumbnail_dir: Path,
                  tile_progress: bool = False) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    slide_metadata: Dict[str, Any] = sample["metadata"]
    keys_to_save = ("slide_id", "tile_id", "image", "label",
                    "tile_x", "tile_y", "occupancy")
    metadata_keys = tuple("slide_" + key for key in slide_metadata)
    csv_columns: Tuple[str, ...] = (*keys_to_save, *metadata_keys)
    print(csv_columns)
    slide_id: str = sample["slide_id"]
    rel_slide_dir = Path(slide_id)
    output_tiles_dir = output_dir / rel_slide_dir
    logging.info(f">>> Slide dir {output_tiles_dir}")
    if is_already_processed(output_tiles_dir):
        logging.info(f">>> Skipping {output_tiles_dir} - already processed")
        return output_tiles_dir
    else:
        output_tiles_dir.mkdir(parents=True, exist_ok=True)
        dataset_csv_path = output_tiles_dir / "dataset.csv"
        dataset_csv_file = dataset_csv_path.open('w', encoding='utf-8')
        dataset_csv_file.write(','.join(csv_columns) + '\n')

        n_failed_tiles = 0
        failed_tiles_csv_path = output_tiles_dir / "failed_tiles.csv"
        failed_tiles_file = failed_tiles_csv_path.open('w', encoding='utf-8')
        failed_tiles_file.write('tile_id' + '\n')

        slide_image_path = Path(sample["image"])
        logging.info(f"Loading slide {slide_id} ...\nFile: {slide_image_path}")

        tmp_dir = tempfile.TemporaryDirectory()
        tmp_slide_image_path = Path(tmp_dir.name) / slide_image_path.name
        try:
            logging.info(f">>> Copying {slide_image_path} to {tmp_slide_image_path}")
            shutil.copy(slide_image_path, tmp_slide_image_path)
            sample["image"] = tmp_slide_image_path
            logging.info(f">>> Finished copying {slide_image_path} to {tmp_slide_image_path}")
        except Exception:
            logging.exception(f"Failed copying slide {slide_image_path} to temporary location")
            # proceed with original path as fallback
            sample["image"] = slide_image_path

        # Save original slide thumbnail (wrapped in try/except)
        try:
            save_thumbnail(slide_image_path, thumbnail_dir / (slide_image_path.name + "_original.png"))
        except Exception:
            logging.exception("save_thumbnail failed")

        loader = LoadROId(WSIReader(backend="OpenSlide"), level=level, margin=margin,
                          foreground_threshold=foreground_threshold)
        # load ROI (may throw)
        sample = loader(sample)  # load 'image' from disk

        # Save ROI thumbnail using PIL (avoid plt)
        try:
            slide_image = sample["image"]  # C,H,W
            img_hwc = np.moveaxis(slide_image, 0, -1).astype(np.uint8)
            pil_roi = Image.fromarray(img_hwc).convert("RGB")
            roi_path = thumbnail_dir / (slide_image_path.name + "_roi.png")
            pil_roi.save(roi_path)
            logging.info(f"Saving ROI thumbnail {roi_path}, shape {pil_roi.size}")
            # explicitly delete pil_roi to avoid destructor in non-main thread
            del pil_roi
        except Exception:
            logging.exception("Failed saving ROI thumbnail via PIL")

        logging.info(f"Tiling slide {slide_id} ...")
        image_tiles, rel_tile_locations, occupancies, _ = \
            generate_tiles(sample["image"], tile_size,
                            sample["foreground_threshold"],
                            occupancy_threshold)

        tile_locations = (sample["scale"] * rel_tile_locations + sample["origin"]).astype(int)  # noqa: W503

        n_tiles = image_tiles.shape[0]
        logging.info(f"{n_tiles} tiles found")

        tile_info_list = []

        logging.info(f"Saving tiles for slide {slide_id} ...")
        for i in tqdm(range(n_tiles), f"Tiles ({slide_id[:6]}…)", unit="img", disable=not tile_progress):
            try:
                tile_info = get_tile_info(sample, occupancies[i], tile_locations[i], rel_slide_dir)
                tile_info_list.append(tile_info)
                try:
                    save_image(image_tiles[i], output_dir / tile_info["image"])
                except Exception:
                    # 记录单个 tile 保存失败，但不要中断整个 slide 的处理
                    logging.exception(f"Failed to save tile image for {tile_info['tile_id']}")
                    n_failed_tiles += 1
                    failed_tiles_file.write(get_tile_descriptor(tile_locations[i]) + '\n')
                    continue

                dataset_row = format_csv_row(tile_info.copy(), keys_to_save, metadata_keys)
                dataset_csv_file.write(dataset_row + '\n')
            except Exception as e:
                n_failed_tiles += 1
                descriptor = get_tile_descriptor(tile_locations[i])
                failed_tiles_file.write(descriptor + '\n')
                logging.exception(f"An error occurred while saving tile {get_tile_id(slide_id, tile_locations[i])}: {e}")
                warnings.warn(f"An error occurred while saving tile {get_tile_id(slide_id, tile_locations[i])}: {e}")

        dataset_csv_file.close()
        failed_tiles_file.close()

        # tile location overlay using PIL (more thread-friendly)
        try:
            visualize_tile_locations_pil(sample, thumbnail_dir / (slide_image_path.name + "_roi_tiles.png"), tile_info_list, tile_size, origin_offset=sample["origin"])
        except Exception:
            logging.exception("visualize_tile_locations_pil failed")

        if n_failed_tiles > 0:
            logging.warning(f"{slide_id} is incomplete. {n_failed_tiles} tiles failed.")

        logging.info(f"Finished processing slide {slide_id}")

        # cleanup temp dir (TemporaryDirectory() will be cleaned up on object deletion)
        try:
            tmp_dir.cleanup()
        except Exception:
            pass

        return output_tiles_dir


def merge_dataset_csv_files(dataset_dir: Path) -> Path:
    full_csv = dataset_dir / "dataset.csv"
    with full_csv.open('w', encoding='utf-8') as full_csv_file:
        first_file = True
        for slide_csv in tqdm(dataset_dir.glob("*/dataset.csv"), desc="Merging dataset.csv", unit='file'):
            logging.info(f"Merging slide {slide_csv}")
            content = slide_csv.read_text(encoding='utf-8')
            if not first_file:
                content = content[content.index('\n') + 1:]
            full_csv_file.write(content)
            first_file = False
    return full_csv


def main(slides_dataset: "SlidesDataset", root_output_dir: Union[str, Path],
         level: int, tile_size: int, margin: int, foreground_threshold: Optional[float],
         occupancy_threshold: float, parallel: bool = False, overwrite: bool = False,
         n_slides: Optional[int] = None) -> None:

    dataset = Dataset(slides_dataset)[:n_slides]  # type: ignore

    for sample in dataset:
        image_path = Path(sample["image_path"])
        assert image_path.exists(), f"{image_path} doesn't exist"

    output_dir = Path(root_output_dir)
    logging.info(f"Creating dataset of level-{level} {tile_size}x{tile_size} "
                 f"{slides_dataset.__class__.__name__} tiles at: {output_dir}")

    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=not overwrite)
    thumbnail_dir = output_dir / "thumbnails"
    thumbnail_dir.mkdir(exist_ok=True)
    logging.info(f"Thumbnail directory: {thumbnail_dir}")

    func = functools.partial(process_slide, level=level, margin=margin, tile_size=tile_size,
                             foreground_threshold=foreground_threshold,
                             occupancy_threshold=occupancy_threshold, output_dir=output_dir,
                             thumbnail_dir=thumbnail_dir,
                             tile_progress=not parallel)

    if parallel:
        import multiprocessing
        pool = multiprocessing.Pool()
        map_func = pool.imap_unordered  # type: ignore
    else:
        map_func = map  # type: ignore

    list(tqdm(map_func(func, dataset), desc="Slides", unit="img", total=len(dataset)))  # type: ignore

    if parallel:
        pool.close()

    logging.info("Merging slide files in a single file")
    merge_dataset_csv_files(output_dir)

# --------------------------------------------------------
# Pipeline for running with GigaPath
# (modified to include an AsyncInferencePipeline implementation)
# --------------------------------------------------------
import os
import timm
import torch
import shutil
import numpy as np
import pandas as pd
import gigapath.slide_encoder as slide_encoder

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from torchvision import transforms
from typing import List, Tuple, Union, Dict, Any
from torch.utils.data import Dataset, DataLoader
from gigapath.preprocessing.data.create_tiles_dataset import process_slide

# --- keep original dataset and functions for backward compatibility ---
class TileEncodingDataset(Dataset):
    """
    Do encoding for tiles

    Arguments:
    ----------
    image_paths : List[str]
        List of image paths, each image is named with its coordinates
        Example: ['images/256x_256y.png', 'images/256x_512y.png']
    transform : torchvision.transforms.Compose
        Transform to apply to each image
    """
    def __init__(self, image_paths: List[str], transform=None):
        self.transform = transform
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)
        # get x, y coordinates from the image name
        x, y = img_name.split('.png')[0].split('_')
        x, y = int(x.replace('x', '')), int(y.replace('y', ''))
        # load the image
        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")
            if self.transform:
                img = self.transform(img)
        return {'img': torch.from_numpy(np.array(img)),
                'coords': torch.from_numpy(np.array([x, y])).float()}


def tile_one_slide(slide_file:str='', save_dir:str='', level:int=0, tile_size:int=256):
    """
    This function is used to tile a single slide and save the tiles to a directory.
    -------------------------------------------------------------------------------
    Warnings: pixman 0.38 has a known bug, which produces partial broken images.
    Make sure to use a different version of pixman.
    -------------------------------------------------------------------------------
    """
    slide_id = os.path.basename(slide_file)
    slide_sample = {"image": slide_file, "slide_id": slide_id, "metadata": {}}

    save_dir = Path(save_dir)
    if save_dir.exists():
        print(f"Warning: Directory {save_dir} already exists. ")

    print(f"Processing slide {slide_file} at level {level} with tile size {tile_size}. Saving to {save_dir}.")

    slide_dir = process_slide(
        slide_sample,
        level=level,
        margin=0,
        tile_size=tile_size,
        foreground_threshold=None,
        occupancy_threshold=0.1,
        output_dir=save_dir / "output",
        thumbnail_dir=save_dir / "thumbnails",
        tile_progress=True,
    )

    dataset_csv_path = slide_dir / "dataset.csv"
    dataset_df = pd.read_csv(dataset_csv_path)
    assert len(dataset_df) > 0
    failed_csv_path = slide_dir / "failed_tiles.csv"
    failed_df = pd.read_csv(failed_csv_path)
    assert len(failed_df) == 0

    print(f"Slide {slide_file} has been tiled. {len(dataset_df)} tiles saved to {slide_dir}.")


def load_tile_encoder_transforms() -> transforms.Compose:
    """Load the transforms for the tile encoder"""
    transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transform


def load_tile_slide_encoder(local_tile_encoder_path: str='',
                            local_slide_encoder_path: str='',
                            global_pool=False) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Load the GigaPath tile and slide encoder models."""
    if local_tile_encoder_path:
        tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False, checkpoint_path=local_tile_encoder_path)
    else:
        tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    print("Tile encoder param #", sum(p.numel() for p in tile_encoder.parameters()))

    if local_slide_encoder_path:
        slide_encoder_model = slide_encoder.create_model(local_slide_encoder_path, "gigapath_slide_enc12l768d", 1536, global_pool=global_pool)
    else:
        slide_encoder_model = slide_encoder.create_model("hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536, global_pool=global_pool)
    print("Slide encoder param #", sum(p.numel() for p in slide_encoder_model.parameters()))

    return tile_encoder, slide_encoder_model


@torch.no_grad()
def run_inference_with_tile_encoder(image_paths: List[str], tile_encoder: torch.nn.Module, batch_size: int=128) -> dict:
    """
    Run inference with the tile encoder (同步接口，保持兼容)
    """
    tile_encoder = tile_encoder.cuda()
    tile_dl = DataLoader(TileEncodingDataset(image_paths, transform=load_tile_encoder_transforms()), batch_size=batch_size, shuffle=False)
    tile_encoder.eval()
    collated_outputs = {'tile_embeds': [], 'coords': []}
    with torch.cuda.amp.autocast(dtype=torch.float16):
        for batch in tqdm(tile_dl, desc='Running inference with tile encoder'):
            collated_outputs['tile_embeds'].append(tile_encoder(batch['img'].cuda()).detach().cpu())
            collated_outputs['coords'].append(batch['coords'])
    return {k: torch.cat(v) for k, v in collated_outputs.items()}


@torch.no_grad()
def run_inference_with_slide_encoder(tile_embeds: torch.Tensor, coords: torch.Tensor, slide_encoder_model: torch.nn.Module) -> torch.Tensor:
    """
    Run inference with the slide encoder (同步接口，保持兼容)
    """
    if len(tile_embeds.shape) == 2:
        tile_embeds = tile_embeds.unsqueeze(0)
        coords = coords.unsqueeze(0)

    slide_encoder_model = slide_encoder_model.cuda()
    slide_encoder_model.eval()
    with torch.cuda.amp.autocast(dtype=torch.float16):
        slide_embeds = slide_encoder_model(tile_embeds.cuda(), coords.cuda(), all_layer_embed=True)
    outputs = {"layer_{}_embed".format(i): slide_embeds[i].cpu() for i in range(len(slide_embeds))}
    outputs["last_layer_embed"] = slide_embeds[-1].cpu()
    return outputs

# --------------------------
# 新增：异步推理流水线实现（单文件集成）
# --------------------------
import time
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty

# --------- 辅助小函数 ----------
def parse_coords_from_filename(fn: str):
    """从 tile 文件名解析坐标，兼容不同命名风格（常见 '256x_512y.png'）"""
    base = os.path.basename(fn)
    name = base.split('.png')[0]
    parts = name.split('_')
    x = y = None
    for p in parts:
        if p.endswith('x'):
            try:
                x = int(p[:-1])
            except:
                pass
        elif p.endswith('y'):
            try:
                y = int(p[:-1])
            except:
                pass
        else:
            if 'x' in p and 'y' in p:
                try:
                    sx = p.split('x')[0]
                    sy = p.split('x')[1].split('y')[0]
                    x = int(sx); y = int(sy)
                except:
                    pass
    if x is None or y is None:
        import re
        nums = re.findall(r'(-?\d+)', name)
        if len(nums) >= 2:
            x, y = int(nums[0]), int(nums[1])
        else:
            raise ValueError(f"无法从文件名解析坐标: {fn}")
    return int(x), int(y)

# --------- GPUTask & SlideAggregator & GPUWorker 类（核心） ----------
class GPUTask:
    def __init__(self, task_type: str, payload: dict, priority: int = 0):
        self.task_type = task_type
        self.payload = payload
        self.priority = priority
        self.enqueued_time = time.time()

class SlideAggregator:
    """
    聚合每张 slide 的 tile embedding；当收到全部 tile 后，提交 slide_aggregate 任务
    slide_tile_count_map: slide_id -> expected tile count
    """
    def __init__(self, slide_tile_count_map: Dict[str, int], gpu_task_queue: Queue, save_dir: str):
        self.slide_tile_count_map = slide_tile_count_map
        self.gpu_task_queue = gpu_task_queue
        self.save_dir = save_dir
        self.lock = threading.Lock()
        self._storage = {}  # slide_id -> {'coords': [...], 'embeds': [...], 'received': int}
        self.total_pending_embeddings = 0

    def register_slide(self, slide_id: str):
        with self.lock:
            if slide_id not in self._storage:
                self._storage[slide_id] = {'coords': [], 'embeds': [], 'received': 0}

    def push_tile_embeddings(self, slide_id: str, coords_np: np.ndarray, tile_embeds_tensor: torch.Tensor):
        with self.lock:
            st = self._storage.setdefault(slide_id, {'coords': [], 'embeds': [], 'received': 0})
            st['coords'].append(coords_np)
            st['embeds'].append(tile_embeds_tensor)
            st['received'] += coords_np.shape[0]
            self.total_pending_embeddings += coords_np.shape[0]

            expected = self.slide_tile_count_map.get(slide_id, None)
            if expected is not None and st['received'] >= expected:
                coords_arr = np.vstack(st['coords']).astype(np.float32)
                embeds = torch.cat(st['embeds'], dim=0)  # cpu tensor
                out_path = os.path.join(self.save_dir, f"{slide_id}_embed.pt")
                payload = {
                    'tile_embeds': embeds,
                    'coords': torch.from_numpy(coords_arr).float(),
                    'slide_id': slide_id,
                    'out_path': out_path,
                }
                task = GPUTask('slide_aggregate', payload, priority=-1)
                # 尝试非阻塞入队，失败则阻塞入队（保证不丢）
                try:
                    self.gpu_task_queue.put(task, block=False)
                except:
                    self.gpu_task_queue.put(task)
                self.total_pending_embeddings -= st['received']
                del self._storage[slide_id]

    def get_total_pending(self):
        with self.lock:
            return self.total_pending_embeddings

class SmartGPUWorker(threading.Thread):
    """
    单个 GPU worker：从 gpu_task_queue 取任务执行（tile_encode / slide_aggregate）
    """
    def __init__(self, gpu_task_queue: Queue, tile_encoder: torch.nn.Module, slide_encoder_model: torch.nn.Module, scheduler_stats: dict):
        super().__init__(daemon=True)
        self.gpu_task_queue = gpu_task_queue
        self.tile_encoder = tile_encoder.cuda()
        self.slide_encoder_model = slide_encoder_model.cuda()
        self.scheduler_stats = scheduler_stats
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.tile_encoder.eval()
        self.slide_encoder_model.eval()

    def run(self):
        while not self.stop_event.is_set():
            try:
                task: GPUTask = self.gpu_task_queue.get(timeout=1.0)
            except Empty:
                with self.lock:
                    self.scheduler_stats['last_idle_ts'] = time.time()
                continue

            start_ts = time.time()
            try:
                if task.task_type == 'tile_encode':
                    payload = task.payload
                    imgs_cpu: torch.Tensor = payload['imgs']       # CPU (B,C,H,W)
                    coords_np: np.ndarray = payload['coords']     # (B,2)
                    slide_tags = payload.get('slide_tags', None)  # list len B
                    B = imgs_cpu.shape[0]
                    chunk_size = 512
                    emb_list = []
                    with torch.no_grad():
                        for i in range(0, B, chunk_size):
                            end = min(B, i + chunk_size)
                            chunk = imgs_cpu[i:end].cuda(non_blocking=True)
                            with torch.cuda.amp.autocast(dtype=torch.float16):
                                embeds_gpu = self.tile_encoder(chunk)
                            emb_list.append(embeds_gpu.detach().cpu())
                    embeds_cpu = torch.cat(emb_list, dim=0)
                    # 按 slide_tags 分割并 push
                    mapping = {}
                    for idx, sid in enumerate(slide_tags):
                        mapping.setdefault(sid, []).append(idx)
                    for sid, indices in mapping.items():
                        subset_emb = embeds_cpu[indices]
                        subset_coords = coords_np[indices]
                        self.scheduler_stats['aggregator'].push_tile_embeddings(sid, subset_coords, subset_emb)
                elif task.task_type == 'slide_aggregate':
                    payload = task.payload
                    tile_embeds_cpu = payload['tile_embeds']
                    coords_cpu = payload['coords']
                    slide_id = payload['slide_id']
                    out_path = payload['out_path']
                    # 使用原有 run_inference_with_slide_encoder 来保持一致输出
                    outputs = run_inference_with_slide_encoder(tile_embeds_cpu, coords_cpu, self.slide_encoder_model)
                    last_layer_embed = outputs["last_layer_embed"]
                    try:
                        torch.save(last_layer_embed, out_path)
                    except Exception as e:
                        print(f"[SmartGPUWorker] 保存 slide embedding 失败 {slide_id}: {e}")
                # 更新统计
                elapsed = time.time() - start_ts
                with self.lock:
                    prev = self.scheduler_stats.get('gpu_busy_ma', 0.0)
                    alpha = 0.2
                    self.scheduler_stats['gpu_busy_ma'] = prev * (1 - alpha) + elapsed * alpha
                    self.scheduler_stats['last_gpu_op_ts'] = time.time()
                    self.scheduler_stats['last_gpu_op_elapsed'] = elapsed
            finally:
                try:
                    self.gpu_task_queue.task_done()
                except:
                    pass

    def stop(self):
        self.stop_event.set()

# --------- Preprocessor & Scheduler ----------
class Preprocessor:
    def __init__(self, transform):
        self.transform = transform

    def preprocess_image(self, image_path: str):
        with open(image_path, "rb") as f:
            img = Image.open(f).convert("RGB")
            t = self.transform(img)
            return t.float()

    def preprocess_batch(self, image_paths: List[str]):
        imgs = []
        coords = []
        for p in image_paths:
            try:
                t = self.preprocess_image(p)
            except Exception as e:
                print(f"[Preprocessor] fail to load {p}: {e}")
                continue
            imgs.append(t)
            x, y = parse_coords_from_filename(p)
            coords.append([x, y])
        if len(imgs) == 0:
            return None, None
        imgs_tensor = torch.stack(imgs, dim=0)
        coords_np = np.asarray(coords, dtype=np.float32)
        return imgs_tensor, coords_np

class Scheduler:
    """
    接收 tile path，构造 microbatch 并投递 GPU 任务（tile_encode）。
    支持简单动态微批（grow/shrink）和 backpressure（基于 aggregator backlog 和 gpu queue）
    """
    def __init__(self, tile_preproc_executor: ThreadPoolExecutor, gpu_task_queue: Queue,
                 slide_tile_count_map: Dict[str, int], aggregator: SlideAggregator, transform,
                 base_microbatch: int = 64,
                 microbatch_min: int = 8, microbatch_max: int = 512,
                 queue_high_water: int = 64, queue_low_water: int = 8,
                 agg_backpressure_threshold: int = 200):
        self.tile_preproc_executor = tile_preproc_executor
        self.gpu_task_queue = gpu_task_queue
        self.slide_tile_count_map = slide_tile_count_map
        self.aggregator = aggregator
        self.transform = transform
        self.microbatch_size = base_microbatch
        self.min_mb = microbatch_min
        self.max_mb = microbatch_max
        self.queue_high_water = queue_high_water
        self.queue_low_water = queue_low_water
        self.agg_backpressure_threshold = agg_backpressure_threshold

        self.pending_tile_paths = Queue()
        self.shutdown_event = threading.Event()
        self.scheduler_stats = {
            'gpu_busy_ma': 0.0,
            'last_gpu_op_ts': time.time(),
            'last_idle_ts': time.time(),
            'last_gpu_op_elapsed': 0.01,
            'aggregator': self.aggregator,
        }
        self._assembler_thread = threading.Thread(target=self._assembler_loop, daemon=True)

    def start(self):
        self._assembler_thread.start()

    def stop(self):
        self.shutdown_event.set()
        self._assembler_thread.join(timeout=5)

    def submit_tile_paths(self, slide_id: str, image_paths: List[str]):
        self.aggregator.register_slide(slide_id)
        for p in image_paths:
            self.pending_tile_paths.put((slide_id, p))

    def _assembler_loop(self):
        preproc = Preprocessor(self.transform)
        while not self.shutdown_event.is_set():
            agg_pending = self.aggregator.get_total_pending()
            gpu_qsize = self.gpu_task_queue.qsize()
            if agg_pending > self.agg_backpressure_threshold or gpu_qsize > self.queue_high_water:
                time.sleep(0.1)
                self._shrink_microbatch()
                continue

            if self.scheduler_stats.get('gpu_busy_ma', 0.0) < 0.05 and gpu_qsize < self.queue_low_water:
                self._grow_microbatch()

            # gather up to microbatch_size tile paths
            bpaths = []
            try:
                for _ in range(self.microbatch_size):
                    sid, p = self.pending_tile_paths.get(timeout=0.5)
                    bpaths.append((sid, p))
            except Empty:
                if len(bpaths) == 0:
                    continue

            grouped = {}
            for sid, path in bpaths:
                grouped.setdefault(sid, []).append(path)

            imgs_list = []
            coords_list = []
            slide_tags = []
            for sid, paths in grouped.items():
                imgs_tensor, coords_np = preproc.preprocess_batch(paths)
                if imgs_tensor is None:
                    continue
                imgs_list.append(imgs_tensor)
                coords_list.append(coords_np)
                slide_tags += [sid] * imgs_tensor.shape[0]

            if len(imgs_list) == 0:
                continue

            imgs_mb = torch.cat(imgs_list, dim=0)
            coords_mb = np.vstack(coords_list)
            task_payload = {
                'imgs': imgs_mb,       # CPU float32
                'coords': coords_mb,   # numpy
                'slide_tags': slide_tags,
            }
            task = GPUTask('tile_encode', task_payload, priority=0)

            while self.gpu_task_queue.qsize() > self.queue_high_water:
                time.sleep(0.05)
                self._shrink_microbatch()
            self.gpu_task_queue.put(task)

    def _grow_microbatch(self):
        new_mb = min(self.max_mb, int(self.microbatch_size * 1.25) + 1)
        if new_mb != self.microbatch_size:
            self.microbatch_size = new_mb

    def _shrink_microbatch(self):
        new_mb = max(self.min_mb, int(self.microbatch_size * 0.7))
        if new_mb != self.microbatch_size:
            self.microbatch_size = new_mb

# --------- AsyncInferencePipeline 封装 ----------
class AsyncInferencePipeline:
    """
    使用方法示例：

    pipeline = AsyncInferencePipeline(tile_encoder, slide_encoder_model,
                                      saved_slide_embed_dir='J:/GigaPath_embed',
                                      tmp_dir_base='J:/Tiles_tmp',
                                      source_dir='G:/LiverDatasets/HE-274',
                                      target_mpp=0.5)

    pipeline.start()   # 启动线程池与 GPU worker

    # 两种使用方式之一：
    # 1) 如果你自己先用 tile_one_slide 生成 tiles：调用 submit_slide_tiles(slide_id, image_paths)
    # 2) 让 pipeline 异步完成 tiling 并提交：调用 submit_slide_for_tiling(slide_path)（会在后台运行 tile_one_slide 并在完成后提交）
    # ...
    # pipeline.wait_until_done()
    # pipeline.stop()
    """
    def __init__(self,
                 tile_encoder: torch.nn.Module,
                 slide_encoder_model: torch.nn.Module,
                 saved_slide_embed_dir: str,
                 tmp_dir_base: str,
                 source_dir: str,
                 target_mpp: float = 0.5,
                 num_preprocess_threads: int = None,
                 tiling_threadpool_size: int = 2,
                 base_microbatch: int = 64):
        self.tile_encoder = tile_encoder
        self.slide_encoder_model = slide_encoder_model
        self.saved_slide_embed_dir = saved_slide_embed_dir
        self.tmp_dir_base = tmp_dir_base
        self.source_dir = source_dir
        self.target_mpp = target_mpp

        self.num_preprocess_threads = num_preprocess_threads or max(4, (os.cpu_count() or 8) - 2)
        self.tiling_threadpool_size = tiling_threadpool_size
        self.base_microbatch = base_microbatch

        os.makedirs(self.saved_slide_embed_dir, exist_ok=True)
        os.makedirs(self.tmp_dir_base, exist_ok=True)

        self.gpu_task_queue = Queue(maxsize=256)
        self.slide_tile_count_map: Dict[str, int] = {}
        self.aggregator = SlideAggregator(self.slide_tile_count_map, self.gpu_task_queue, self.saved_slide_embed_dir)
        self.transform = load_tile_encoder_transforms()

        self.tile_preproc_executor = ThreadPoolExecutor(max_workers=self.num_preprocess_threads)
        self.tiling_executor = ThreadPoolExecutor(max_workers=self.tiling_threadpool_size)

        self.scheduler = Scheduler(self.tile_preproc_executor, self.gpu_task_queue, self.slide_tile_count_map, self.aggregator, self.transform, base_microbatch=self.base_microbatch)
        self.gpu_worker = SmartGPUWorker(self.gpu_task_queue, self.tile_encoder, self.slide_encoder_model, self.scheduler.scheduler_stats)

        self._tiling_futures = {}  # future -> (slide_id, slide_path, tmp_dir)

    def start(self):
        print("[AsyncInferencePipeline] start scheduler and gpu worker")
        self.scheduler.start()
        self.gpu_worker.start()

    def stop(self):
        print("[AsyncInferencePipeline] stopping ...")
        # 等待 tiling futures 完成（可改为超时或取消）
        for fut in list(self._tiling_futures.keys()):
            try:
                fut.result(timeout=10)
            except Exception:
                pass
        # 等待 pending -> submit 完成
        while not self.scheduler.pending_tile_paths.empty():
            time.sleep(0.2)
        # 等待 gpu tasks 完成
        self.gpu_task_queue.join()
        # 等待 aggregator backlog 清空
        while self.aggregator.get_total_pending() > 0:
            time.sleep(0.2)
        self.scheduler.stop()
        self.gpu_worker.stop()
        self.gpu_worker.join(timeout=5)
        self.tile_preproc_executor.shutdown(wait=True)
        self.tiling_executor.shutdown(wait=True)
        print("[AsyncInferencePipeline] stopped.")

    def submit_slide_tiles(self, slide_id: str, image_paths: List[str]):
        """
        当 tiles 已由外部生成完毕时可调用（同步方式）
        """
        self.slide_tile_count_map[slide_id] = len(image_paths)
        self.scheduler.submit_tile_paths(slide_id, image_paths)
        print(f"[AsyncInferencePipeline] Submitted {len(image_paths)} tiles of slide {slide_id} to scheduler.")

    def submit_slide_for_tiling(self, slide_path: str, tile_size: int = 256):
        """
        异步地执行 tile_one_slide，并在完成后把 tile paths 自动 submit 给 scheduler
        """
        slide_id = os.path.splitext(os.path.basename(slide_path))[0]
        tmp_dir = os.path.join(self.tmp_dir_base, slide_id)
        fut = self.tiling_executor.submit(self._safe_tile_one_slide_and_submit, slide_path, tmp_dir, tile_size)
        self._tiling_futures[fut] = (slide_id, slide_path, tmp_dir)

    def _safe_tile_one_slide_and_submit(self, slide_path: str, tmp_dir: str, tile_size: int = 256):
        os.makedirs(tmp_dir, exist_ok=True)
        level = self._get_level_for_mpp(slide_path, self.target_mpp)
        if level is None:
            raise ValueError(f"No pyramid level matches {self.target_mpp} MPP for slide {slide_path}")
        tile_one_slide(slide_file=slide_path, save_dir=tmp_dir, level=level, tile_size=tile_size)
        slide_dir = os.path.join(tmp_dir, "output", os.path.basename(slide_path))
        if not os.path.exists(slide_dir):
            # fallback: sometimes process_slide stores differently
            slide_dir = os.path.join(tmp_dir, "output")
        image_paths = [os.path.join(slide_dir, x) for x in os.listdir(slide_dir) if x.lower().endswith('.png')]
        if len(image_paths) == 0:
            print(f"[AsyncInferencePipeline] slide {slide_path} produced 0 tiles.")
            return
        # update expected tile count and submit
        self.slide_tile_count_map[os.path.splitext(os.path.basename(slide_path))[0]] = len(image_paths)
        self.scheduler.submit_tile_paths(os.path.splitext(os.path.basename(slide_path))[0], image_paths)
        print(f"[AsyncInferencePipeline] Tiling done and submitted {len(image_paths)} tiles for slide {slide_path}")

    def _get_level_for_mpp(self, slide_path, target_mpp=0.5):
        try:
            from gigapath.preprocessing.data.slide_utils import find_level_for_target_mpp as _find
            lvl = _find(slide_path, target_mpp)
            if lvl is not None:
                return lvl
        except Exception:
            pass
        slide = None
        try:
            slide = Image.open(slide_path) if slide_path.lower().endswith(('.png', '.jpg')) else None
        except:
            slide = None
        # fallback to openslide metadata
        try:
            import openslide
            slide_os = openslide.OpenSlide(slide_path)
            mpp_x = float(slide_os.properties.get('openslide.mpp-x', 0) or 0)
            downsamples = slide_os.level_downsamples
            slide_os.close()
            if mpp_x <= 0:
                return None
            diffs = [abs(mpp_x * d - target_mpp) for d in downsamples]
            return int(np.argmin(diffs))
        except Exception:
            return None

    def wait_until_done(self, timeout: float = None):
        """
        简单等待直到没有 pending tiles / gpu queue / aggregator backlog。可指定 timeout（秒）。
        """
        start = time.time()
        while True:
            if timeout is not None and (time.time() - start) > timeout:
                break
            pending = not self.scheduler.pending_tile_paths.empty()
            qsize = self.gpu_task_queue.qsize()
            backlog = self.aggregator.get_total_pending()
            if (not pending) and qsize == 0 and backlog == 0:
                break
            time.sleep(0.2)
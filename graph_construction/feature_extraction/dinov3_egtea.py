from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf
from tqdm import tqdm
from transformers import AutoModel

from graph_construction.feature_extraction.sampling import uniform_sample_frames


REPO_ROOT = Path("/home/s3758869/egocentric_video_graph_framework_ar")
DEFAULT_VIDEOS_ROOT = REPO_ROOT / "data/videos"
DEFAULT_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_MODEL_NAME = "dinov3_vitl16"

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    hf_token = "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"
    
@dataclass(frozen=True)
class ClipRecord:
    clip_dir: Path
    clip_name: str
    frames: tuple[Path, ...]
    output_path: Path
    total_available_frames: int

class EgteaFrameDataset(Dataset):
    def __init__(self, clips: list[ClipRecord], image_size: int) -> None:
        self.clips = clips
        self.image_size = image_size
        self.offsets: list[int] = []
        total = 0
        for clip in clips:
            total += len(clip.frames)
            self.offsets.append(total)

        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return self.offsets[-1] if self.offsets else 0

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int, str]:
        clip_idx = bisect.bisect_right(self.offsets, index)
        prev = 0 if clip_idx == 0 else self.offsets[clip_idx - 1]
        frame_idx = index - prev
        frame_path = self.clips[clip_idx].frames[frame_idx]

        image = read_image(str(frame_path), mode=ImageReadMode.RGB).to(torch.float32).div_(255.0)
        image = tvf.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        image = (image - self.mean) / self.std
        return image, clip_idx, frame_idx, frame_path.name

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-root", type=Path, default=DEFAULT_VIDEOS_ROOT)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--pooling", choices=("mean", "cls", "pooler"), default="mean")
    parser.add_argument("--save-dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--limit-clips", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--max-frames-per-clip", type=int, default=32)

    return parser.parse_args()

def slurm_shard_defaults(args: argparse.Namespace) -> tuple[int, int]:
    if args.num_shards is not None and args.shard_index is not None:
        return args.num_shards, args.shard_index

    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    task_count = os.environ.get("SLURM_ARRAY_TASK_COUNT")
    task_min = int(os.environ.get("SLURM_ARRAY_TASK_MIN", "0"))
    if task_id is not None and task_count is not None:
        return int(task_count), int(task_id) - task_min

    return args.num_shards or 1, args.shard_index or 0

def default_split_roots(videos_root: Path) -> list[Path]:
    roots: list[Path] = []
    for split_name, prefix in (("train", "train_split_"), ("test", "test_split_")):
        split_base = videos_root / split_name
        for split_id in (1, 2, 3):
            roots.append(split_base / f"{prefix}{split_id}")
    return roots


def collect_clips(args: argparse.Namespace, num_shards: int, shard_index: int) -> list[ClipRecord]:
    if args.splits:
        split_roots = [Path(split) for split in args.splits]
    else:
        split_roots = default_split_roots(args.videos_root)

    by_real_path: dict[Path, Path] = {}
    for split_root in split_roots:
        if not split_root.exists():
            print(f"Skipping missing split root: {split_root}", flush=True)
            continue
        for child in split_root.iterdir():
            if child.is_symlink():
                target = os.readlink(child)
                real_path = Path(target) if os.path.isabs(target) else (child.parent / target)
                by_real_path[real_path] = child
            elif child.is_dir():
                by_real_path[child.resolve()] = child

    real_clip_dirs = sorted(by_real_path)
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"--shard-index must be in [0, {num_shards})")
    real_clip_dirs = [
        clip_dir for i, clip_dir in enumerate(real_clip_dirs) if i % num_shards == shard_index
    ]
    if args.limit_clips is not None:
        real_clip_dirs = real_clip_dirs[: args.limit_clips]

    clips: list[ClipRecord] = []
    output_name = f"frame_features_model_{args.model_name}.h5"
    for clip_dir in real_clip_dirs:
        output_path = clip_dir / output_name
        if output_path.exists() and not args.overwrite:
            continue
        all_frames = tuple(sorted((clip_dir / "frames").glob("*.jpg")))
        frames = uniform_sample_frames(all_frames, args.max_frames_per_clip)
        if frames:
            clips.append(
                ClipRecord(
                    clip_dir=clip_dir,
                    clip_name=clip_dir.name,
                    frames=frames,
                    output_path=output_path,
                    total_available_frames=len(all_frames),
                )
            )
    return clips


def load_csv_by_frame(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {row["frame_file"]: row for row in csv.DictReader(f)}


def to_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def to_int(value: str | None) -> int:
    if value is None or value == "":
        return -1
    try:
        return int(float(value))
    except ValueError:
        return -1


def write_clip_h5(
    clip: ClipRecord,
    frame_names: list[str],
    features: list[np.ndarray],
    args: argparse.Namespace,
) -> None:
    order = np.argsort(np.array([int(Path(name).stem) for name in frame_names], dtype=np.int64))
    sorted_names = [frame_names[i] for i in order]
    visual_features = np.stack([features[i] for i in order], axis=0).astype(args.save_dtype)

    frame_metadata = load_csv_by_frame(clip.clip_dir / "frame_metadata.csv")
    annotations = load_csv_by_frame(clip.clip_dir / "annotations.csv")

    frame_ids = np.array([int(Path(name).stem) for name in sorted_names], dtype=np.int32)
    frame_strings = np.array(sorted_names, dtype=h5py.string_dtype(encoding="utf-8"))
    source_frame_index = np.array(
        [to_int(frame_metadata.get(name, {}).get("source_frame_index")) for name in sorted_names],
        dtype=np.int32,
    )
    egtea_frame_number = np.array(
        [to_int(frame_metadata.get(name, {}).get("egtea_frame_number")) for name in sorted_names],
        dtype=np.int32,
    )

    clip.output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{clip.output_path.name}.", suffix=".tmp", dir=clip.output_path.parent
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("frame_ids", data=frame_ids)
        f.create_dataset("frame_names", data=frame_strings)
        f.create_dataset("visual_features", data=visual_features)
        f.create_dataset("clip_feature", data=visual_features.mean(axis=0))
        f.create_dataset("source_frame_index", data=source_frame_index)
        f.create_dataset("egtea_frame_number", data=egtea_frame_number)

        if annotations:
            gaze = np.array(
                [
                    [
                        to_float(annotations.get(name, {}).get("gaze_x")),
                        to_float(annotations.get(name, {}).get("gaze_y")),
                    ]
                    for name in sorted_names
                ],
                dtype=np.float32,
            )
            f.create_dataset("gaze_labels", data=gaze)
            for key in ("action_id", "verb_id", "activity_block_id"):
                f.create_dataset(
                    key,
                    data=np.array(
                        [to_int(annotations.get(name, {}).get(key)) for name in sorted_names],
                        dtype=np.int32,
                    ),
                )
            for key in ("action", "activity", "noun_ids"):
                values = np.array(
                    [annotations.get(name, {}).get(key, "") for name in sorted_names],
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
                f.create_dataset(key, data=values)

        f.attrs["clip_name"] = clip.clip_name
        f.attrs["model_id"] = args.model_id
        f.attrs["model_name"] = args.model_name
        f.attrs["pooling"] = args.pooling
        f.attrs["image_size"] = args.image_size
        f.attrs["save_dtype"] = args.save_dtype
        f.attrs["num_frames"] = len(sorted_names)
        f.attrs["max_frames_per_clip"] = args.max_frames_per_clip
        f.attrs["sampled_num_frames"] = len(sorted_names)

        metadata_path = clip.clip_dir / "clip_metadata.json"
        if metadata_path.exists():
            f.attrs["clip_metadata_json"] = json.dumps(json.loads(metadata_path.read_text()))
    os.replace(tmp_path, clip.output_path)

    if tmp_path.exists():
        tmp_path.unlink()


def extract_features(model: torch.nn.Module, pixel_values: torch.Tensor, pooling: str) -> torch.Tensor:
    outputs = model(pixel_values=pixel_values)

    if pooling == "pooler" and getattr(outputs, "pooler_output", None) is not None:
        feats = outputs.pooler_output
    else:
        tokens = outputs.last_hidden_state
        if pooling == "cls" or pooling == "pooler":
            feats = tokens[:, 0]
        else:
            feats = tokens.mean(dim=1)

    if not torch.isfinite(feats).all():
        raise RuntimeError("Model produced NaN/Inf features")

    return feats


def main() -> None:
    args = parse_args()
    num_shards, shard_index = slurm_shard_defaults(args)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    clips = collect_clips(args, num_shards=num_shards, shard_index=shard_index)
    total_frames = sum(len(clip.frames) for clip in clips)
    print(
        f"EGTEA DINOv3 extraction: {len(clips)} clips, {total_frames} frames, "
        f"shard {shard_index + 1}/{num_shards}",
        flush=True,
    )
    if not clips:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model_id} on {device}", flush=True)
    model = AutoModel.from_pretrained(
        args.model_id,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        local_files_only=args.local_files_only,
        token=hf_token,
        torch_dtype=torch.float16 if device.type == "cuda" and not args.no_amp else torch.float32,
    )
    model.eval().to(device)
    if args.compile:
        model = torch.compile(model)

    dataset = EgteaFrameDataset(clips, image_size=args.image_size)
    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)

    expected = {i: len(clip.frames) for i, clip in enumerate(clips)}
    frame_names: dict[int, list[str]] = {}
    feature_buffers: dict[int, list[np.ndarray]] = {}
    completed = 0
    started = time.time()

    amp_enabled = device.type == "cuda" and not args.no_amp
    with torch.inference_mode():
        progress = tqdm(total=total_frames, desc="DINOv3 frames", dynamic_ncols=True)
        for images, clip_indices, _, names in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                feats = extract_features(model, images, args.pooling)
            feats_np = feats.detach().float().cpu().numpy()

            for clip_idx, name, feat in zip(clip_indices.tolist(), names, feats_np, strict=True):
                frame_names.setdefault(clip_idx, []).append(name)
                feature_buffers.setdefault(clip_idx, []).append(feat)
                if len(feature_buffers[clip_idx]) == expected[clip_idx]:
                    write_clip_h5(clips[clip_idx], frame_names[clip_idx], feature_buffers[clip_idx], args)
                    del frame_names[clip_idx]
                    del feature_buffers[clip_idx]
                    completed += 1
                    if completed % 100 == 0:
                        elapsed = max(time.time() - started, 1e-6)
                        print(
                            f"Completed {completed}/{len(clips)} clips "
                            f"({progress.n / elapsed:.1f} frames/s)",
                            flush=True,
                        )
            progress.update(len(names))
        progress.close()

    elapsed = max(time.time() - started, 1e-6)
    print(
        f"Finished {completed}/{len(clips)} clips, {total_frames} frames "
        f"in {elapsed / 60:.1f} min ({total_frames / elapsed:.1f} frames/s)",
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract GroundingDINO hand bbox features for EGTEA Gaze+ hand masks.

The hand masks are treated as annotations: a frame is processed only when a
non-empty hand mask is available, and each connected hand-mask component is
matched to exactly one GroundingDINO hand prediction.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.ops import box_convert
from tqdm import tqdm


DEFAULT_ANNOTATION_ROOT = Path(
    "/home/s3758869/egocentric_video_graph_framework_ar/data/egtea_gaze_plus/"
    "vlm_ann_Qwen3-VL-32B-Instruct"
)
DEFAULT_FRAMES_ROOT = Path(
    "/projects/eemcs/dmb/ComputerVision/ego_graphs/vlm_datasets/egtea_gaze/"
    "framewise_videos"
)
DEFAULT_HAND_MASK_ROOT = Path(
    "/projects/eemcs/dmb/ComputerVision/ego_graphs/vlm_datasets/egtea_gaze/hand_masks"
)

tools_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools")
sys.path.insert(0, tools_path)

import groundingdino.datasets.transforms as T  # noqa: E402
from groundingdino.models import build_model  # noqa: E402
from groundingdino.util.slconfig import SLConfig  # noqa: E402
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap  # noqa: E402


def load_model(model_config_path, model_checkpoint_path, device="cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model.to(device)


def load_image(image_path: Path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    return image_pil, image


def get_model_variant(config_file: str) -> str:
    config_name = os.path.basename(config_file).lower()
    return "gdino_base" if "swinb" in config_name else "gdino_tiny"


def slurm_shard_defaults(args) -> tuple[int, int]:
    if args.num_shards is not None and args.shard_index is not None:
        return args.num_shards, args.shard_index

    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    task_count = os.environ.get("SLURM_ARRAY_TASK_COUNT")
    task_min = int(os.environ.get("SLURM_ARRAY_TASK_MIN", "0"))
    if task_id is not None and task_count is not None:
        return int(task_count), int(task_id) - task_min
    return args.num_shards or 1, args.shard_index or 0


def discover_split_roots(annotation_root: Path, requested_splits: list[str] | None) -> list[Path]:
    if requested_splits:
        roots = []
        for split in requested_splits:
            split_path = Path(split).expanduser()
            roots.append(split_path if split_path.is_absolute() else annotation_root / split_path)
        return roots

    roots = []
    for mode in ("train", "test"):
        mode_dir = annotation_root / mode
        if not mode_dir.exists():
            continue
        roots.extend(sorted(path for path in mode_dir.iterdir() if path.is_dir()))
    return roots


def split_rel_path(annotation_root: Path, split_root: Path) -> Path:
    try:
        return split_root.resolve().relative_to(annotation_root.resolve())
    except ValueError:
        return Path(split_root.name)


def read_source_clip_dir(annotation_clip_path: Path) -> Path | None:
    metadata_path = annotation_clip_path / "vlm_annotation_metadata.json"
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    source_clip_dir = metadata.get("source_clip_dir")
    return Path(source_clip_dir) if source_clip_dir else None


def resolve_frames_dir(annotation_root: Path, frames_root: Path, annotation_clip_path: Path) -> Path | None:
    local_frames = annotation_clip_path / "frames"
    if local_frames.exists():
        return local_frames

    rel_clip = annotation_clip_path.relative_to(annotation_root)
    split_frames = frames_root / rel_clip / "frames"
    if split_frames.exists():
        return split_frames

    source_clip_dir = read_source_clip_dir(annotation_clip_path)
    if source_clip_dir is not None and (source_clip_dir / "frames").exists():
        return source_clip_dir / "frames"
    return None


def output_is_complete(output_path: Path, expected_frames: int) -> bool:
    if not output_path.exists():
        return False
    try:
        with output_path.open("rb") as f:
            data = pickle.load(f)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    meta = data.get("__metadata__", {})
    return meta.get("annotation_frames") == expected_frames


def collect_annotation_clips(args, model_variant: str) -> list[tuple[Path, Path]]:
    annotation_root = args.annotation_root
    num_shards, shard_index = slurm_shard_defaults(args)
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"--shard-index must be in [0, {num_shards})")

    candidates = []
    if args.clip_list is not None:
        with args.clip_list.open("r", encoding="utf-8") as f:
            clip_paths = [
                Path(line.strip()).expanduser()
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            ]
        for clip_path in sorted(clip_paths):
            if not clip_path.is_absolute():
                clip_path = annotation_root / clip_path
            if (clip_path / args.annotation_name).exists():
                candidates.append((split_rel_path(annotation_root, clip_path.parent), clip_path))
    else:
        for split_root in discover_split_roots(annotation_root, args.splits):
            if not split_root.exists():
                print(f"Skipping missing split root: {split_root}", flush=True)
                continue
            rel = split_rel_path(annotation_root, split_root)
            for clip_path in sorted(path for path in split_root.iterdir() if path.is_dir()):
                if (clip_path / args.annotation_name).exists():
                    candidates.append((rel, clip_path))

    candidates = [item for index, item in enumerate(candidates) if index % num_shards == shard_index]
    if args.limit_clips:
        candidates = candidates[: args.limit_clips]

    selected = []
    for rel, clip_path in candidates:
        output_path = clip_path / f"hand_grounding_results_{model_variant}.pkl"
        if not args.overwrite:
            try:
                expected_frames = len(pd.read_csv(clip_path / args.annotation_name))
            except Exception:
                expected_frames = 0
            if output_is_complete(output_path, expected_frames):
                continue
        selected.append((rel, clip_path))
    return selected


def session_name_from_clip(clip_name: str) -> str:
    parts = clip_name.split("-")
    if len(parts) < 3:
        return clip_name
    return "-".join(parts[:3])


def hand_mask_candidates(session_name: str, row, hand_mask_fps: float) -> list[int]:
    candidates = []
    for key in ("hand_mask_frame", "hand_mask_frame_number"):
        if key in row and not pd.isna(row[key]):
            candidates.append(int(row[key]))
    if "egtea_frame_number" in row and not pd.isna(row["egtea_frame_number"]):
        egtea_frame = int(row["egtea_frame_number"])
        candidates.extend(
            [
                round(egtea_frame / hand_mask_fps),
                math.floor(egtea_frame / hand_mask_fps),
                math.ceil(egtea_frame / hand_mask_fps),
                egtea_frame,
            ]
        )
    if "source_frame_index" in row and not pd.isna(row["source_frame_index"]):
        source_frame = int(row["source_frame_index"])
        candidates.extend([source_frame, source_frame + 1])

    deduped = []
    for candidate in candidates:
        if candidate >= 0 and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def resolve_mask_path(args, session_name: str, row) -> Path | None:
    masks_dir = args.hand_mask_root / "Masks"
    for frame_number in hand_mask_candidates(session_name, row, args.hand_mask_fps):
        mask_path = masks_dir / f"{session_name}_{frame_number:06d}.png"
        if mask_path.exists():
            return mask_path
    return None


def connected_components(mask: np.ndarray, min_area: int) -> list[dict[str, object]]:
    binary = mask > 0
    if not binary.any():
        return []

    try:
        import cv2

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary.astype(np.uint8), connectivity=8
        )
        components = []
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label].tolist()
            if area < min_area:
                continue
            components.append(
                {
                    "component_id": len(components),
                    "bbox_xyxy": [float(x), float(y), float(x + w), float(y + h)],
                    "area": int(area),
                }
            )
        return components
    except Exception:
        ys, xs = np.where(binary)
        area = int(binary.sum())
        if area < min_area:
            return []
        return [
            {
                "component_id": 0,
                "bbox_xyxy": [
                    float(xs.min()),
                    float(ys.min()),
                    float(xs.max() + 1),
                    float(ys.max() + 1),
                ],
                "area": area,
            }
        ]


def load_mask_components(mask_path: Path, image_size: tuple[int, int], min_area: int) -> list[dict[str, object]]:
    mask = np.array(Image.open(mask_path).convert("L"))
    components = connected_components(mask, min_area)
    if not components:
        return []

    mask_h, mask_w = mask.shape[:2]
    img_w, img_h = image_size
    scale_x = img_w / mask_w
    scale_y = img_h / mask_h
    for component in components:
        x1, y1, x2, y2 = component["bbox_xyxy"]
        component["bbox_xyxy_mask"] = [x1, y1, x2, y2]
        component["bbox_xyxy"] = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
        component["mask_size"] = [int(mask_w), int(mask_h)]
    return components


def filter_predictions(outputs, box_threshold, text_threshold, tokenizer, caption):
    logits = outputs["pred_logits"].sigmoid()[0].cpu()
    boxes = outputs["pred_boxes"][0].cpu()
    features = outputs["bounding_boxes_features"][0].cpu()

    tokenized = tokenizer(caption)
    phrases = [
        get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).strip()
        for logit in logits
    ]
    scores = logits.max(dim=1)[0]
    keep = scores > box_threshold
    if keep.any():
        return boxes[keep], features[keep], logits[keep], scores[keep], [p for p, k in zip(phrases, keep) if bool(k)]
    return boxes, features, logits, scores, phrases


def box_iou_xyxy(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(box_a[:, None, :2], box_b[None, :, :2])
    rb = torch.minimum(box_a[:, None, 2:], box_b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = ((box_a[:, 2] - box_a[:, 0]).clamp(min=0) * (box_a[:, 3] - box_a[:, 1]).clamp(min=0))
    area_b = ((box_b[:, 2] - box_b[:, 0]).clamp(min=0) * (box_b[:, 3] - box_b[:, 1]).clamp(min=0))
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


def match_predictions_to_masks(
    components: list[dict[str, object]],
    boxes_cxcywh: torch.Tensor,
    features: torch.Tensor,
    scores: torch.Tensor,
    phrases: list[str],
    image_size: tuple[int, int],
) -> list[dict[str, object]]:
    if not components or len(boxes_cxcywh) == 0:
        return []

    img_w, img_h = image_size
    boxes_xyxy_norm = box_convert(boxes_cxcywh, in_fmt="cxcywh", out_fmt="xyxy")
    scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=boxes_xyxy_norm.dtype)
    pred_xyxy = boxes_xyxy_norm * scale
    mask_xyxy = torch.tensor([c["bbox_xyxy"] for c in components], dtype=pred_xyxy.dtype)
    ious = box_iou_xyxy(mask_xyxy, pred_xyxy)

    used_predictions = set()
    hands = []
    for mask_idx, component in enumerate(components):
        ranking = sorted(
            range(len(pred_xyxy)),
            key=lambda pred_idx: (float(ious[mask_idx, pred_idx]), float(scores[pred_idx])),
            reverse=True,
        )
        pred_idx = next((idx for idx in ranking if idx not in used_predictions), ranking[0])
        used_predictions.add(pred_idx)
        hands.append(
            {
                "component_id": component["component_id"],
                "mask_area": component["area"],
                "mask_size": component["mask_size"],
                "mask_bbox_xyxy": component["bbox_xyxy"],
                "mask_bbox_xyxy_mask_space": component["bbox_xyxy_mask"],
                "bbox_xyxy": pred_xyxy[pred_idx],
                "bbox_cxcywh_norm": boxes_cxcywh[pred_idx],
                "feats": features[pred_idx],
                "confidence": float(scores[pred_idx]),
                "iou_with_mask_bbox": float(ious[mask_idx, pred_idx]),
                "phrase": phrases[pred_idx],
            }
        )
    return hands


def process_clip(args, model, model_variant: str, clip_path: Path) -> dict[str, int]:
    annotation_path = clip_path / args.annotation_name
    frames_dir = resolve_frames_dir(args.annotation_root, args.frames_root, clip_path)
    if frames_dir is None:
        print(f"Frames not found for {clip_path}, skipping", flush=True)
        return Counter({"clips_missing_frames": 1})

    annotation_rows = pd.read_csv(annotation_path)
    session_name = session_name_from_clip(clip_path.name)
    output = {"__metadata__": {"clip_name": clip_path.name, "session_name": session_name}}
    stats = Counter()

    for _, row in annotation_rows.iterrows():
        stats["annotation_frames"] += 1
        frame_file = str(row.get("frame_file") or f"{int(row['frame_index']):06d}.jpg")
        image_path = frames_dir / frame_file
        if not image_path.exists():
            stats["missing_images"] += 1
            continue

        mask_path = resolve_mask_path(args, session_name, row)
        if mask_path is None:
            stats["missing_mask_files"] += 1
            if args.save_empty_frames:
                output[frame_file] = {"hands": [], "mask_path": None}
            continue

        image_pil, image = load_image(image_path)
        components = load_mask_components(mask_path, image_pil.size, args.min_mask_area)
        if not components:
            stats["empty_masks"] += 1
            if args.save_empty_frames:
                output[frame_file] = {"hands": [], "mask_path": str(mask_path)}
            continue

        with torch.no_grad():
            outputs = model(image.to(args.device)[None], captions=[args.caption])

        boxes, features, logits, scores, phrases = filter_predictions(
            outputs,
            args.box_threshold,
            args.text_threshold,
            model.tokenizer,
            args.caption,
        )
        hands = match_predictions_to_masks(components, boxes, features, scores, phrases, image_pil.size)
        output[frame_file] = {
            "hands": hands,
            "mask_path": str(mask_path),
            "image_path": str(image_path),
            "source_frame_index": int(row["source_frame_index"]) if "source_frame_index" in row else None,
            "egtea_frame_number": int(row["egtea_frame_number"]) if "egtea_frame_number" in row else None,
        }
        stats["frames"] += 1
        stats["hands"] += len(hands)

    output["__metadata__"].update(
        {
            "model_variant": model_variant,
            "caption": args.caption,
            "processed_frames": stats["frames"],
            "annotation_frames": stats["annotation_frames"],
            "hands": stats["hands"],
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "hand_mask_fps": args.hand_mask_fps,
        }
    )
    output_path = clip_path / f"hand_grounding_results_{model_variant}.pkl"
    with output_path.open("wb") as f:
        pickle.dump(output, f)
    return stats


def inference(args) -> None:
    model_variant = get_model_variant(args.config_file)
    print(f"Detected model variant: {model_variant}", flush=True)

    clips = collect_annotation_clips(args, model_variant)
    print(f"EGTEA hand grounding: {len(clips)} clips queued", flush=True)
    if not clips:
        return

    model = load_model(args.config_file, args.model_checkpoint_path, args.device)
    logging.getLogger("GroundingDINO").info("Model loaded successfully")

    totals = Counter()
    for rel, clip_path in tqdm(clips, desc="EGTEA hand clips"):
        print(f"Processing {rel}/{clip_path.name}", flush=True)
        stats = process_clip(args, model, model_variant, clip_path)
        totals.update(stats)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        "completed: annotation_frames={annotation_frames} frames={frames} hands={hands} missing_mask_files={missing_mask_files} "
        "empty_masks={empty_masks} missing_images={missing_images}".format(
            annotation_frames=totals["annotation_frames"],
            frames=totals["frames"],
            hands=totals["hands"],
            missing_mask_files=totals["missing_mask_files"],
            empty_masks=totals["empty_masks"],
            missing_images=totals["missing_images"],
        ),
        flush=True,
    )


def get_args_parser():
    parser = argparse.ArgumentParser("GroundingDINO EGTEA hand feature extraction", add_help=False)
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument("--model_checkpoint_path", "-p", type=str, required=True)
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--frames-root", type=Path, default=DEFAULT_FRAMES_ROOT)
    parser.add_argument("--hand-mask-root", type=Path, default=DEFAULT_HAND_MASK_ROOT)
    parser.add_argument("--annotation-name", default="annotations_qwen3vl_32b_instruct.csv")
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--clip-list", type=Path, default=None)
    parser.add_argument("--limit-clips", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-empty-frames", action="store_true")
    parser.add_argument("--caption", default="hand .")
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.15)
    parser.add_argument("--min-mask-area", type=int, default=64)
    parser.add_argument(
        "--hand-mask-fps",
        type=float,
        default=50.0,
        help="Convert egtea_frame_number to hand-mask frame number via round(frame/fps).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "GroundingDINO EGTEA hand feature extraction", parents=[get_args_parser()]
    )
    parsed_args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    inference(parsed_args)

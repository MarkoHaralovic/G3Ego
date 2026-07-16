#!/usr/bin/env python3
"""Run GroundingDINO over EGTEA VLM annotations."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import math
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert, nms
from tqdm import tqdm


DEFAULT_ANNOTATION_ROOT = Path(
    "/path/to/ego_graphs/vlm_datasets/egtea_gaze/"
    "vlm_ann_Qwen3-VL-32B-Instruct"
)
DEFAULT_FRAMES_ROOT = Path(
    "/path/to/ego_graphs/vlm_datasets/egtea_gaze/framewise_videos"
)
DEFAULT_HAND_MASK_ROOT = Path(
    "/path/to/ego_graphs/vlm_datasets/egtea_gaze/hand_masks"
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
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
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


def parse_objects(raw_value: object) -> dict[str, dict[str, object]]:
    if raw_value is None or (isinstance(raw_value, float) and math.isnan(raw_value)):
        return {}
    if isinstance(raw_value, dict):
        return raw_value
    text = str(raw_value).strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def prepare_caption(frame_objects_dict: dict[str, dict[str, object]]) -> str:
    return (" . ".join(frame_objects_dict.keys()) + ".").lower()


def filter_predictions(outputs, box_threshold, text_threshold, tokenizer, caption, apply_nms=False):
    logits = outputs["pred_logits"].sigmoid()[0]
    boxes = outputs["pred_boxes"][0]
    boxes_features = outputs["bounding_boxes_features"][0]

    logits_filt = logits.cpu().clone()
    boxes_filt = boxes.cpu().clone()
    boxes_features_filt = boxes_features.cpu().clone()

    bbox_threshold_filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[bbox_threshold_filt_mask]
    boxes_filt = boxes_filt[bbox_threshold_filt_mask]
    boxes_features_filt = boxes_features_filt[bbox_threshold_filt_mask]

    tokenized = tokenizer(caption)
    pred_phrases = [
        get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).strip()
        for logit in logits_filt
    ]

    if apply_nms and len(boxes_filt) > 0:
        scores = logits_filt.max(dim=1)[0]
        boxes_xyxy = box_convert(boxes_filt, in_fmt="cxcywh", out_fmt="xyxy")
        nms_keep = nms(boxes_xyxy, scores, iou_threshold=0.5)
        boxes_filt = boxes_filt[nms_keep]
        logits_filt = logits_filt[nms_keep]
        boxes_features_filt = boxes_features_filt[nms_keep]
        pred_phrases = [pred_phrases[i] for i in nms_keep.tolist()]

    return boxes_filt, pred_phrases, boxes_features_filt, logits_filt


def merge_predictions_of_same_class(boxes_filt, pred_phrases, boxes_features_filt, logits_filt):
    phrase_to_features = {}
    for phrase in set(pred_phrases):
        phrase_indexes = [i for i, pred_phrase in enumerate(pred_phrases) if pred_phrase == phrase]
        phrase_indexes_t = torch.tensor(phrase_indexes)
        bbox_feature_per_class = boxes_features_filt[phrase_indexes_t].mean(dim=0)
        phrase_logits = logits_filt[phrase_indexes_t]
        best_idx = phrase_logits.max(dim=1)[0].argmax()
        phrase_bbox = boxes_filt[phrase_indexes_t][best_idx]
        best_confidence = phrase_logits[best_idx].max().item()
        phrase_to_features[phrase] = (phrase_bbox, bbox_feature_per_class, best_confidence)
    return phrase_to_features


def point_to_bbox_distance(x, y, x_min, y_min, x_max, y_max):
    closest_x = max(x_min, min(x, x_max))
    closest_y = max(y_min, min(y, y_max))
    dx = x - closest_x
    dy = y - closest_y
    return math.sqrt(dx * dx + dy * dy)


def bbox_to_xyxy_abs(bbox, image_size) -> list[float]:
    img_w, img_h = image_size
    cx, cy, bw, bh = torch.as_tensor(bbox).cpu().float().tolist()
    return [
        (cx - bw / 2) * img_w,
        (cy - bh / 2) * img_h,
        (cx + bw / 2) * img_w,
        (cy + bh / 2) * img_h,
    ]


def bbox_payload(bbox, image_size) -> dict[str, list[float]]:
    cxcywh = torch.as_tensor(bbox).cpu().float()
    return {
        "bbox_cxcywh_norm": cxcywh.tolist(),
        "bbox_xyxy_norm": box_convert(cxcywh[None], in_fmt="cxcywh", out_fmt="xyxy")[0].tolist(),
        "bbox_xyxy": bbox_to_xyxy_abs(cxcywh, image_size),
    }


def gazed_at_object(boxes_filt, pred_phrases, boxes_features_filt, gaze_x, gaze_y, image_size):
    if gaze_x is None or gaze_y is None:
        return None, None, None, None

    img_w, img_h = image_size
    distances_object_gaze = []
    for bbox in boxes_filt:
        cx, cy, bw, bh = bbox.cpu().numpy()
        x_min = (cx - bw / 2) * img_w
        y_min = (cy - bh / 2) * img_h
        x_max = (cx + bw / 2) * img_w
        y_max = (cy + bh / 2) * img_h
        distances_object_gaze.append(point_to_bbox_distance(gaze_x, gaze_y, x_min, y_min, x_max, y_max))

    if not distances_object_gaze:
        return None, None, None, None
    min_idx = distances_object_gaze.index(min(distances_object_gaze))
    return pred_phrases[min_idx], boxes_features_filt[min_idx], boxes_filt[min_idx], min_idx


def session_name_from_clip(clip_name: str) -> str:
    parts = clip_name.split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else clip_name


def hand_mask_candidates(row, hand_mask_fps: float) -> list[int]:
    candidates = []
    for key in ("hand_mask_frame", "hand_mask_frame_number"):
        if key in row and not pd.isna(row[key]):
            candidates.append(int(row[key]))
    if "egtea_frame_number" in row and not pd.isna(row["egtea_frame_number"]):
        frame = int(row["egtea_frame_number"])
        candidates.extend([round(frame / hand_mask_fps), math.floor(frame / hand_mask_fps), math.ceil(frame / hand_mask_fps), frame])
    if "source_frame_index" in row and not pd.isna(row["source_frame_index"]):
        frame = int(row["source_frame_index"])
        candidates.extend([frame, frame + 1])
    return list(dict.fromkeys(c for c in candidates if c >= 0))


def resolve_mask_path(args, session_name: str, row) -> Path | None:
    masks_dir = args.hand_mask_root / "Masks"
    for frame_number in hand_mask_candidates(row, args.hand_mask_fps):
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

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
        components = []
        for _, (x, y, w, h, area) in enumerate(stats[1:].tolist()):
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
        return [] if area < min_area else [
            {
                "component_id": 0,
                "bbox_xyxy": [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
                "area": area,
            }
        ]


def load_mask_components(mask_path: Path, image_size: tuple[int, int], min_area: int) -> list[dict[str, object]]:
    mask = np.array(Image.open(mask_path).convert("L"))
    components = connected_components(mask, min_area)
    mask_h, mask_w = mask.shape[:2]
    img_w, img_h = image_size
    for component in components:
        x1, y1, x2, y2 = component["bbox_xyxy"]
        component["bbox_xyxy_mask"] = [x1, y1, x2, y2]
        component["bbox_xyxy"] = [x1 * img_w / mask_w, y1 * img_h / mask_h, x2 * img_w / mask_w, y2 * img_h / mask_h]
        component["mask_size"] = [int(mask_w), int(mask_h)]
    return components


def box_iou_xyxy(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(box_a[:, None, :2], box_b[None, :, :2])
    rb = torch.minimum(box_a[:, None, 2:], box_b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = ((box_a[:, 2] - box_a[:, 0]).clamp(min=0) * (box_a[:, 3] - box_a[:, 1]).clamp(min=0))
    area_b = ((box_b[:, 2] - box_b[:, 0]).clamp(min=0) * (box_b[:, 3] - box_b[:, 1]).clamp(min=0))
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp(min=1e-6)


def match_predictions_to_masks(components, boxes_cxcywh, features, scores, phrases, image_size):
    if not components or len(boxes_cxcywh) == 0:
        return []
    img_w, img_h = image_size
    pred_xyxy = box_convert(boxes_cxcywh.cpu(), in_fmt="cxcywh", out_fmt="xyxy") * torch.tensor([img_w, img_h, img_w, img_h])
    mask_xyxy = torch.tensor([c["bbox_xyxy"] for c in components], dtype=pred_xyxy.dtype)
    ious = box_iou_xyxy(mask_xyxy, pred_xyxy)
    used, hands = set(), []
    for mask_idx, component in enumerate(components):
        ranking = sorted(range(len(pred_xyxy)), key=lambda i: (float(ious[mask_idx, i]), float(scores[i])), reverse=True)
        pred_idx = next((i for i in ranking if i not in used), ranking[0])
        used.add(pred_idx)
        hands.append(
            {
                "component_id": component["component_id"],
                "mask_area": component["area"],
                "mask_size": component["mask_size"],
                "mask_bbox_xyxy": component["bbox_xyxy"],
                "mask_bbox_xyxy_mask_space": component["bbox_xyxy_mask"],
                "bbox_xyxy": pred_xyxy[pred_idx],
                "bbox_cxcywh_norm": boxes_cxcywh[pred_idx].cpu(),
                "feats": features[pred_idx].cpu(),
                "confidence": float(scores[pred_idx]),
                "iou_with_mask_bbox": float(ious[mask_idx, pred_idx]),
                "phrase": phrases[pred_idx],
            }
        )
    return hands


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


def frame_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 0, path.name


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
    frame_count = len([key for key in data.keys() if key != "__metadata__"])
    return frame_count == expected_frames


def collect_annotation_clips(args, model_variant: str) -> list[tuple[Path, Path, Path]]:
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
            parse_path = clip_path / args.parse_name
            ann_path = clip_path / args.annotation_name
            if args.only_manual_replacements and not has_manual_replacements(ann_path):
                continue
            if parse_path.exists() and ann_path.exists():
                rel = split_rel_path(annotation_root, clip_path.parent)
                candidates.append((rel, clip_path, parse_path))
            else:
                print(f"Skipping incomplete clip-list entry: {clip_path}", flush=True)
    else:
        split_roots = discover_split_roots(annotation_root, args.splits)
        for split_root in split_roots:
            if not split_root.exists():
                print(f"Skipping missing split root: {split_root}", flush=True)
                continue
            rel = split_rel_path(annotation_root, split_root)
            for clip_path in sorted(path for path in split_root.iterdir() if path.is_dir()):
                parse_path = clip_path / args.parse_name
                ann_path = clip_path / args.annotation_name
                if args.only_manual_replacements and not has_manual_replacements(ann_path):
                    continue
                if parse_path.exists() and ann_path.exists():
                    candidates.append((rel, clip_path, parse_path))

    candidates = [item for index, item in enumerate(candidates) if index % num_shards == shard_index]
    if args.limit_clips:
        candidates = candidates[: args.limit_clips]

    selected = []
    for rel, clip_path, parse_path in candidates:
        output_path = clip_path / f"grounding_results_{model_variant}.pkl"
        hand_output_path = clip_path / f"hand_grounding_results_{model_variant}.pkl"
        if not args.overwrite:
            try:
                expected_frames = len(pd.read_csv(parse_path))
            except Exception:
                expected_frames = 0
            output_done = output_is_complete(output_path, expected_frames)
            hand_done = (
                not args.extract_hands
                or not args.write_hand_output
                or output_is_complete(hand_output_path, expected_frames)
            )
            if output_done and hand_done:
                continue
        selected.append((rel, clip_path, parse_path))
    return selected


def has_manual_replacements(annotation_path: Path) -> bool:
    if not annotation_path.exists():
        return False
    try:
        with annotation_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if (
                reader.fieldnames is None
                or "original_action_before_manual_fix" not in reader.fieldnames
            ):
                return False
            return any(
                (row.get("original_action_before_manual_fix") or "").strip()
                for row in reader
            )
    except Exception:
        return False


def collect_object_maps(annotation_root: Path, parse_name: str) -> tuple[dict[str, int], Counter[str]]:
    occurrences: Counter[str] = Counter()
    for parse_path in annotation_root.rglob(parse_name):
        try:
            frame_rows = pd.read_csv(parse_path, usecols=["all_objects"])
        except Exception:
            continue
        for raw_objects in frame_rows["all_objects"]:
            for obj_info in parse_objects(raw_objects).values():
                base_object = obj_info.get("base_object")
                if base_object:
                    occurrences[str(base_object)] += 1

    object_mapping = {obj: idx for idx, obj in enumerate(sorted(occurrences.keys()))}
    return object_mapping, occurrences


def load_or_build_objects(args) -> dict[str, int]:
    objects_json = args.objects_json or args.annotation_root / "objects.json"
    if objects_json.exists():
        with objects_json.open("r", encoding="utf-8") as f:
            return json.load(f)
    if not args.build_objects:
        raise FileNotFoundError(f"objects.json not found: {objects_json}")

    object_mapping, occurrences = collect_object_maps(args.annotation_root, args.parse_name)
    if not object_mapping:
        raise FileNotFoundError(
            f"Could not build object map; no all_objects found under {args.annotation_root}"
        )
    objects_json.parent.mkdir(parents=True, exist_ok=True)
    with objects_json.open("w", encoding="utf-8") as f:
        json.dump(object_mapping, f, indent=2)
    with (objects_json.parent / "objects_occurrences.json").open("w", encoding="utf-8") as f:
        json.dump(dict(sorted(occurrences.items())), f, indent=2)
    print(f"Built global object map: {objects_json} ({len(object_mapping)} objects)", flush=True)
    return object_mapping


def safe_float(value: object) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def model_forward(model, image, caption: str, args):
    with torch.inference_mode():
        if args.amp and str(args.device).startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return model(image[None], captions=[caption])
        return model(image[None], captions=[caption])


def extract_hands_for_frame(args, model, image, image_size, session_name, row):
    mask_path = resolve_mask_path(args, session_name, row)
    if mask_path is None:
        return [], None
    components = load_mask_components(mask_path, image_size, args.min_mask_area)
    if not components:
        return [], str(mask_path)

    outputs = model_forward(model, image, args.hand_caption, args)
    boxes, phrases, features, logits = filter_predictions(
        outputs,
        args.hand_box_threshold,
        args.hand_text_threshold,
        model.tokenizer,
        args.hand_caption,
        False,
    )
    scores = logits.max(dim=1)[0] if len(logits) else torch.empty(0)
    return match_predictions_to_masks(components, boxes, features, scores, phrases, image_size), str(mask_path)


def detect_objects_for_frame(args, model, image, caption: str | None):
    if caption is None:
        return {}, [], torch.empty(0, 4), torch.empty(0)

    outputs = model_forward(model, image, caption, args)
    boxes, phrases, features, logits = filter_predictions(
        outputs,
        args.box_threshold,
        args.text_threshold,
        model.tokenizer,
        caption,
        args.apply_nms,
    )
    merged = merge_predictions_of_same_class(boxes, phrases, features, logits)
    merged_phrases = list(merged.keys())
    if not merged_phrases:
        return merged, [], boxes[:0], features[:0]
    return (
        merged,
        merged_phrases,
        torch.stack([merged[p][0] for p in merged_phrases]),
        torch.stack([merged[p][1] for p in merged_phrases]),
    )


def build_object_payloads(image_all_objects, merged, object_mapping, image_size):
    payloads = {}
    merged_lower = {key.lower(): value for key, value in merged.items()}
    for obj_name, obj_info in image_all_objects.items():
        obj_idx = object_mapping.get(str(obj_info.get("base_object")))
        if obj_idx is None:
            continue
        match = merged_lower.get(obj_name.lower())
        if match is None:
            payloads[obj_idx] = {"feats": None, "phrase": obj_name, "confidence": 0.0}
            continue
        bbox, feat, conf = match
        payloads[obj_idx] = {
            "feats": feat,
            "phrase": obj_name,
            "confidence": conf,
            **bbox_payload(bbox, image_size),
        }
    return payloads


def resolve_gazed_object_idx(gazed_phrase, image_all_objects, object_mapping):
    if gazed_phrase is None:
        return None
    for ann_name, ann_info in image_all_objects.items():
        if ann_name.lower() == gazed_phrase.lower():
            return object_mapping.get(str(ann_info.get("base_object")))
    return None


def optional_int(row, key):
    return int(row[key]) if key in row and not pd.isna(row[key]) else None


def process_clip(args, model, model_variant: str, object_mapping: dict[str, int], clip_path: Path) -> dict[str, int]:
    parse_path = clip_path / args.parse_name
    annotation_path = clip_path / args.annotation_name
    frames_dir = resolve_frames_dir(args.annotation_root, args.frames_root, clip_path)
    if frames_dir is None:
        print(f"Frames not found for {clip_path}, skipping", flush=True)
        return {"frames": 0, "hands": 0, "missing_gaze": 0, "missing_images": 0, "missing_masks": 0}

    parsed_annotations = pd.read_csv(parse_path)
    annotation_rows = pd.read_csv(annotation_path)
    annotation_by_frame = {
        str(row["frame_file"]): row for _, row in annotation_rows.iterrows() if "frame_file" in row
    }

    session_name = session_name_from_clip(clip_path.name)
    groundings = {"__metadata__": {"clip_name": clip_path.name, "model_variant": model_variant}}
    hand_groundings = {"__metadata__": {"clip_name": clip_path.name, "session_name": session_name, "model_variant": model_variant}}
    stats = {"frames": 0, "hands": 0, "missing_gaze": 0, "missing_images": 0, "missing_masks": 0}
    for _, ann_row in parsed_annotations.iterrows():
        frame_file = str(ann_row.get("frame_file") or f"{int(ann_row['frame_id']):06d}.jpg")
        image_path = frames_dir / frame_file
        if not image_path.exists():
            stats["missing_images"] += 1
            continue

        image_all_objects = parse_objects(ann_row.get("all_objects"))
        if not image_all_objects and not args.extract_hands:
            continue
        caption = prepare_caption(image_all_objects) if image_all_objects else None

        gaze_row = annotation_by_frame.get(frame_file, ann_row)
        gaze_x = safe_float(gaze_row.get("gaze_x"))
        gaze_y = safe_float(gaze_row.get("gaze_y"))
        if gaze_x is None or gaze_y is None:
            stats["missing_gaze"] += 1

        image_pil, image = load_image(image_path)
        image = image.to(args.device)

        merged, merged_phrases, merged_boxes, merged_features = detect_objects_for_frame(
            args, model, image, caption
        )
        gazed_phrase, gazed_bbox_feature, gazed_bbox, _ = gazed_at_object(
            merged_boxes, merged_phrases, merged_features, gaze_x, gaze_y, image_pil.size
        )
        gazed_idx = resolve_gazed_object_idx(gazed_phrase, image_all_objects, object_mapping)
        objects_dict = build_object_payloads(
            image_all_objects, merged, object_mapping, image_pil.size
        )

        hands, mask_path = ([], None)
        if args.extract_hands:
            hands, mask_path = extract_hands_for_frame(args, model, image, image_pil.size, session_name, gaze_row)
            stats["hands"] += len(hands)
            if mask_path is None:
                stats["missing_masks"] += 1
            if args.write_hand_output:
                hand_groundings[frame_file] = {
                    "hands": hands,
                    "mask_path": mask_path,
                    "image_path": str(image_path),
                    "source_frame_index": optional_int(gaze_row, "source_frame_index"),
                    "egtea_frame_number": optional_int(gaze_row, "egtea_frame_number"),
                }

        groundings[frame_file] = {
            "objects": objects_dict,
            "object_gazed_at": {
                "feats": gazed_bbox_feature,
                "phrase": gazed_phrase,
                "idx": gazed_idx,
                **(bbox_payload(gazed_bbox, image_pil.size) if gazed_bbox is not None else {}),
            },
            "gaze": {"x": gaze_x, "y": gaze_y, "valid": gaze_x is not None and gaze_y is not None},
        }
        if args.extract_hands:
            groundings[frame_file]["hands"] = hands
            groundings[frame_file]["hand_mask_path"] = mask_path
        stats["frames"] += 1

    groundings["__metadata__"].update(stats)
    output_path = clip_path / f"grounding_results_{model_variant}.pkl"
    with output_path.open("wb") as f:
        pickle.dump(groundings, f)
    if args.extract_hands and args.write_hand_output:
        hand_groundings["__metadata__"].update(stats)
        with (clip_path / f"hand_grounding_results_{model_variant}.pkl").open("wb") as f:
            pickle.dump(hand_groundings, f)
    return stats


def inference(args) -> None:
    logger = logging.getLogger("GroundingDINO")
    model_variant = get_model_variant(args.config_file)
    print(f"Detected model variant: {model_variant}", flush=True)

    object_mapping = load_or_build_objects(args)
    print(f"Using global object map with {len(object_mapping)} objects", flush=True)

    clips = collect_annotation_clips(args, model_variant)
    print(f"EGTEA visual grounding: {len(clips)} clips queued", flush=True)
    if not clips:
        return

    model = load_model(args.config_file, args.model_checkpoint_path, args.device)
    logger.info("Model loaded successfully")

    totals = Counter()
    for rel, clip_path, _ in tqdm(clips, desc="EGTEA clips"):
        print(f"Processing {rel}/{clip_path.name}", flush=True)
        stats = process_clip(args, model, model_variant, object_mapping, clip_path)
        totals.update(stats)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        "completed: frames={frames} hands={hands} missing_gaze={missing_gaze} missing_masks={missing_masks} missing_images={missing_images}".format(
            frames=totals["frames"],
            hands=totals["hands"],
            missing_gaze=totals["missing_gaze"],
            missing_masks=totals["missing_masks"],
            missing_images=totals["missing_images"],
        ),
        flush=True,
    )


def get_args_parser():
    parser = argparse.ArgumentParser("GroundingDINO EGTEA inference", add_help=False)
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument("--model_checkpoint_path", "-p", type=str, required=True)
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--frames-root", type=Path, default=DEFAULT_FRAMES_ROOT)
    parser.add_argument("--hand-mask-root", type=Path, default=DEFAULT_HAND_MASK_ROOT)
    parser.add_argument("--objects-json", type=Path, default=None)
    parser.add_argument("--annotation-name", default="annotations_qwen3vl_32b_instruct.csv")
    parser.add_argument("--parse-name", default="parse_annotation.csv")
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument(
        "--clip-list",
        type=Path,
        default=None,
        help="Text file of clip directories to process, absolute or relative to annotation root.",
    )
    parser.add_argument("--limit-clips", type=int, default=0)
    parser.add_argument(
        "--only-manual-replacements",
        action="store_true",
        help="Only process clips whose annotation CSV contains manual action-text replacements.",
    )
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.15)
    parser.add_argument("--apply-nms", action="store_true")
    parser.add_argument("--extract-hands", action="store_true")
    parser.add_argument("--write-hand-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hand-caption", default="hand .")
    parser.add_argument("--hand_box_threshold", type=float, default=0.25)
    parser.add_argument("--hand_text_threshold", type=float, default=0.15)
    parser.add_argument("--min-mask-area", type=int, default=64)
    parser.add_argument("--hand-mask-fps", type=float, default=50.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-build-objects", dest="build_objects", action="store_false")
    parser.set_defaults(build_objects=True)
    parser.add_argument("--device", type=str, default="cuda")
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser("GroundingDINO EGTEA Inference", parents=[get_args_parser()])
    parsed_args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    inference(parsed_args)

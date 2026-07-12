from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import torch
import torch.multiprocessing as mp
import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph_construction.feature_extraction.sampling import uniform_sample_frames  # noqa: E402


@dataclass(frozen=True)
class ClipRecord:
    clip_dir: Path
    split_rel_paths: tuple[Path, ...]
    clip_name: str
    frames: tuple[Path, ...]
    output_clip_paths: tuple[Path, ...]
    total_available_frames: int


def as_path(cfg: dict[str, Any], key: str) -> Path:
    return Path(cfg[key]).expanduser()


def default_split_roots(videos_root: Path) -> list[Path]:
    return [
        videos_root / mode / f"{prefix}{split_id}"
        for mode, prefix in (("train", "train_split_"), ("test", "test_split_"))
        for split_id in (1, 2, 3)
    ]


def configured_split_roots(cfg: dict[str, Any]) -> list[Path]:
    videos_root = as_path(cfg, "videos_root")
    splits = cfg.get("splits")
    if not splits:
        return default_split_roots(videos_root)
    paths = [Path(split).expanduser() for split in splits]
    return [path if path.is_absolute() else videos_root / path for path in paths]


def split_rel_path(videos_root: Path, split_root: Path) -> Path:
    try:
        return split_root.resolve().relative_to(videos_root.resolve())
    except ValueError:
        return Path(split_root.name)


def sort_frame_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 0, path.name


def slurm_shard_defaults(cfg: dict[str, Any]) -> tuple[int, int]:
    if cfg.get("num_shards") is not None and cfg.get("shard_index") is not None:
        return int(cfg["num_shards"]), int(cfg["shard_index"])

    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    task_count = os.environ.get("SLURM_ARRAY_TASK_COUNT")
    task_min = int(os.environ.get("SLURM_ARRAY_TASK_MIN", "0"))
    if task_id is not None and task_count is not None:
        return int(task_count), int(task_id) - task_min

    return int(cfg.get("num_shards") or 1), int(cfg.get("shard_index") or 0)


def load_csv_by_frame(path: Path) -> dict[str, dict[str, str]]:
    rows = load_csv_rows(path) or []
    return {row["frame_file"]: row for row in rows}


def load_csv_rows(path: Path) -> list[dict[str, str]] | None:
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return None


def load_actions(output_path: Path, frame_names: tuple[str, ...], partial=False) -> list[str] | None:
    rows = load_csv_rows(output_path)
    if rows is None:
        return None
    row_frame_names = tuple(row.get("frame_file", "") for row in rows)
    expected_names = frame_names[: len(rows)] if partial else frame_names
    if row_frame_names != expected_names:
        return None
    if partial and len(rows) <= len(frame_names):
        return [row.get("action", "") for row in rows]
    if not partial and len(rows) == len(frame_names):
        return [row.get("action", "") for row in rows]
    return None


def output_is_complete(output_path: Path, frame_names: tuple[str, ...]) -> bool:
    return load_actions(output_path, frame_names) is not None


def atomic_write(path: Path, writer) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    writer(tmp_path)
    tmp_path.replace(path)


def write_lines_atomic(path: Path, lines: list[str]) -> None:
    atomic_write(path, lambda tmp: tmp.write_text("\n".join(lines) + "\n", encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    def writer(tmp_path: Path) -> None:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    atomic_write(path, writer)


def write_records_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    atomic_write(path, lambda tmp: pd.DataFrame(records).to_csv(tmp, index=False))


def collect_clips(cfg: dict[str, Any]) -> list[ClipRecord]:
    videos_root = as_path(cfg, "videos_root")
    output_root = as_path(cfg, "output_data_folder")
    max_frames = int(cfg.get("max_frames_per_clip", 32))
    overwrite = bool(cfg.get("overwrite", False))
    num_shards, shard_index = slurm_shard_defaults(cfg)

    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards})")

    by_real_path: dict[Path, tuple[Path, list[Path]]] = {}
    for split_root in configured_split_roots(cfg):
        if not split_root.exists():
            print(f"Skipping missing split root: {split_root}", flush=True)
            continue
        rel = split_rel_path(videos_root, split_root)
        for child in sorted(split_root.iterdir(), key=lambda p: p.name):
            if child.is_dir():
                real_path = child.resolve()
                if real_path not in by_real_path:
                    by_real_path[real_path] = (child, [])
                by_real_path[real_path][1].append(rel)

    candidates = [
        item
        for i, item in enumerate(sorted(by_real_path.items()))
        if i % num_shards == shard_index
    ]
    if cfg.get("limit_clips") is not None:
        candidates = candidates[: int(cfg["limit_clips"])]

    clips: list[ClipRecord] = []
    for real_clip_dir, (display_clip_dir, rels) in candidates:
        all_frames = tuple(
            sorted((real_clip_dir / "frames").glob("*.jpg"), key=sort_frame_key)
        )
        frames = uniform_sample_frames(all_frames, max_frames)
        if not frames:
            continue

        output_clip_paths = tuple(
            output_root / rel / display_clip_dir.name for rel in sorted(set(rels))
        )
        frame_names = tuple(path.name for path in frames)
        if not overwrite and all(
            output_is_complete(path / f"annotations_{cfg['model_id']}.csv", frame_names)
            for path in output_clip_paths
        ):
            continue

        clips.append(
            ClipRecord(
                clip_dir=real_clip_dir,
                split_rel_paths=tuple(sorted(set(rels))),
                clip_name=display_clip_dir.name,
                frames=frames,
                output_clip_paths=output_clip_paths,
                total_available_frames=len(all_frames),
            )
        )
    return clips


def read_image_rgb(image_path: Path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read frame: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_model(cfg: dict[str, Any], vlm, device: str):
    return vlm.load_model(
        cfg["model_path"],
        device,
        four_bit=cfg["four_bit_quantization"],
        eight_bit=cfg["eight_bit_quantization"],
        model_cache_dir=cfg["model_cache_dir"],
    )


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    with (SCRIPT_DIR / cfg["prompts_file"]).open("r", encoding="utf-8") as f:
        prompts = json.load(f)
    cfg["action_recognition_prompt"] = prompts[cfg["action_recognition_prompt"]]
    return cfg


def action_annotate(
    cfg,
    vlm,
    model,
    processor,
    image_paths: tuple[Path, ...],
    *,
    start_index: int = 0,
    total_frames: int | None = None,
    on_action=None,
) -> list[str]:
    image_size = int(cfg["image_size"])
    actions: list[str] = []
    progress = tqdm.tqdm(
        image_paths,
        desc="Action annotating clip",
        total=total_frames or len(image_paths),
        initial=start_index,
    )
    for i, image_path in enumerate(progress, start=start_index):
        image = read_image_rgb(image_path)
        result = vlm.recognize_action_single_frame(
            model,
            processor,
            image,
            cfg["action_recognition_prompt"],
            image_size,
            max_new_tokens=cfg.get("max_new_tokens_action", 100),
        )
        action = vlm.parse_output(result)
        actions.append(action)
        if on_action is not None:
            on_action(action)
        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
    return actions


def merged_annotation_record(
    sample_index: int,
    frame_path: Path,
    action: str,
    source_rows: dict[str, dict[str, str]],
    metadata_rows: dict[str, dict[str, str]],
    model_id: str,
) -> dict[str, Any]:
    frame_name = frame_path.name
    source = source_rows.get(frame_name, {})
    metadata = metadata_rows.get(frame_name, {})
    pick = lambda key, default="": source.get(key, metadata.get(key, default))
    return {
        "sample_index": sample_index,
        "frame_index": pick("frame_index", frame_path.stem),
        "frame_file": frame_name,
        "source_frame_index": pick("source_frame_index"),
        "egtea_frame_number": pick("egtea_frame_number"),
        "action": action,
        "source_action": source.get("action", ""),
        "source_activity": source.get("activity", ""),
        **{key: source.get(key, "") for key in (
            "activity_block_id", "action_id", "verb_id", "noun_ids", "gaze_x", "gaze_y"
        )},
        "model_id": model_id,
    }


def save_annotations(
    clip: ClipRecord,
    actions: list[str],
    cfg: dict[str, Any],
) -> None:
    source_rows = load_csv_by_frame(clip.clip_dir / "annotations.csv")
    metadata_rows = load_csv_by_frame(clip.clip_dir / "frame_metadata.csv")
    annotated_frames = clip.frames[: len(actions)]

    records = [
        merged_annotation_record(
            i,
            frame_path,
            actions[i] if i < len(actions) else "",
            source_rows,
            metadata_rows,
            cfg["model_id"],
        )
        for i, frame_path in enumerate(annotated_frames)
    ]

    metadata = {
        "clip_name": clip.clip_name,
        "source_clip_dir": str(clip.clip_dir),
        "split_rel_paths": [str(path) for path in clip.split_rel_paths],
        "total_available_frames": clip.total_available_frames,
        "sampled_frames": len(clip.frames),
        "annotated_frames": len(actions),
        "complete": len(actions) == len(clip.frames),
        "max_frames_per_clip": int(cfg.get("max_frames_per_clip", 32)),
        "sampling": "uniform_sample_frames from graph_construction.feature_extraction.sampling",
        "model_id": cfg["model_id"],
        "model_name": cfg.get("model_name", ""),
    }

    for output_clip_path in clip.output_clip_paths:
        output_clip_path.mkdir(parents=True, exist_ok=True)
        write_lines_atomic(output_clip_path / "actions.txt", actions)
        write_records_atomic(output_clip_path / f"annotations_{cfg['model_id']}.csv", records)
        write_json_atomic(output_clip_path / "vlm_annotation_metadata.json", metadata)


def existing_actions_for_clip(clip: ClipRecord, cfg: dict[str, Any]) -> tuple[list[str], bool]:
    frame_names = tuple(path.name for path in clip.frames)
    longest_partial: list[str] = []
    for output_clip_path in clip.output_clip_paths:
        annotations_path = output_clip_path / f"annotations_{cfg['model_id']}.csv"
        actions = load_actions(annotations_path, frame_names)
        if actions is not None:
            return actions, True
        partial = load_actions(annotations_path, frame_names, partial=True) or []
        if len(partial) > len(longest_partial):
            longest_partial = partial
    return longest_partial, False


def annotate_clip(cfg, vlm, model, processor, clip: ClipRecord) -> None:
    actions: list[str] = []
    if not bool(cfg.get("overwrite", False)):
        actions, complete = existing_actions_for_clip(clip, cfg)
        if complete:
            save_annotations(clip, actions, cfg)
            return
        if actions:
            print(
                f"Resuming {clip.clip_name}: {len(actions)}/{len(clip.frames)} frames already annotated",
                flush=True,
            )
            save_annotations(clip, actions, cfg)

    start_index = len(actions)
    if start_index >= len(clip.frames):
        save_annotations(clip, actions, cfg)
        return

    def checkpoint(action: str) -> None:
        actions.append(action)
        save_annotations(clip, actions, cfg)

    action_annotate(
        cfg,
        vlm,
        model,
        processor,
        clip.frames[start_index:],
        start_index=start_index,
        total_frames=len(clip.frames),
        on_action=checkpoint,
    )


def clips_to_process(cfg: dict[str, Any]) -> list[ClipRecord]:
    clips = collect_clips(cfg)
    total_frames = sum(len(clip.frames) for clip in clips)
    print(
        f"EGTEA VLM annotation: {len(clips)} clips, {total_frames} sampled frames",
        flush=True,
    )
    return clips


def annotate_dataset(cfg, vlm, model, processor) -> None:
    clips = clips_to_process(cfg)
    for clip in tqdm.tqdm(clips, desc="Processing EGTEA clips"):
        annotate_clip(cfg, vlm, model, processor, clip)


def gpu_worker(gpu_id, task_queue, result_queue, cfg, vlm) -> None:
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    print(f"[GPU {gpu_id}] Loading model...", flush=True)
    model, processor = load_model(cfg, vlm, device)
    print(f"[GPU {gpu_id}] Model loaded on {model.device}", flush=True)

    while True:
        task = task_queue.get()
        if task is None:
            break
        clip = task
        try:
            first_rel = clip.split_rel_paths[0] if clip.split_rel_paths else Path(".")
            print(f"[GPU {gpu_id}] Processing {first_rel}/{clip.clip_name}", flush=True)
            annotate_clip(cfg, vlm, model, processor, clip)
            result_queue.put((str(first_rel / clip.clip_name), "success", None))
        except Exception as exc:
            result_queue.put((clip.clip_name, "error", str(exc)))


def annotate_dataset_mp(cfg, vlm, num_gpus: int) -> None:
    clips = clips_to_process(cfg)
    if not clips:
        return

    task_queue = mp.Queue()
    result_queue = mp.Queue()
    for task in (*clips, *([None] * num_gpus)):
        task_queue.put(task)

    workers = [
        mp.Process(
            target=gpu_worker,
            args=(gpu_id, task_queue, result_queue, cfg, vlm),
        )
        for gpu_id in range(num_gpus)
    ]
    for process in workers:
        process.start()

    completed = 0
    errors = []
    with tqdm.tqdm(total=len(clips), desc="Total progress") as pbar:
        while completed < len(clips):
            clip_name, status, error = result_queue.get()
            completed += 1
            pbar.update(1)
            if status == "error":
                errors.append((clip_name, error))
                pbar.set_postfix({"errors": len(errors)})

    for process in workers:
        process.join()

    if errors:
        print("Errors:", flush=True)
        for clip_name, error in errors:
            print(f"  - {clip_name}: {error}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser("EGTEA Gaze+ VLM Annotation Pipeline")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to JSON config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    vlm = importlib.import_module(cfg["vlm_module"])
    as_path(cfg, "output_data_folder").mkdir(parents=True, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    quant = cfg["four_bit_quantization"] or cfg["eight_bit_quantization"]

    if num_gpus == 1 or quant:
        device = "cuda:0" if num_gpus == 1 else "cuda"
        model, processor = load_model(cfg, vlm, device)
        print(model.device, flush=True)
        annotate_dataset(cfg, vlm, model, processor)
    elif num_gpus > 1:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        annotate_dataset_mp(cfg, vlm, num_gpus)
    else:
        raise RuntimeError("No GPU detected.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Populate EGTEA annotation CSV gaze columns from BeGaze text exports."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GAZE_DIRS = [
    REPO_ROOT / "data" / "gaze" / "gaze_data" / "gaze_data",
    Path("/projects/eemcs/dmb/ComputerVision/ego_graphs/vlm_datasets/egtea_gaze/gaze/gaze_data/gaze_data"),
]
DEFAULT_ANNOTATION_ROOTS = [
    Path("/projects/eemcs/dmb/ComputerVision/ego_graphs/vlm_datasets/egtea_gaze/framewise_videos/frame_cache_fps_30"),
    Path("/projects/eemcs/dmb/ComputerVision/ego_graphs/vlm_datasets/egtea_gaze/vlm_ann_Qwen3-VL-32B-Instruct"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gaze-dir",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument(
        "--annotation-root",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument(
        "--pattern",
        default="annotations*.csv",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--no-interpolate",
        action="store_true",
        help="Only use frames explicitly present in the BeGaze export.",
    )
    parser.add_argument(
        "--edge-fill",
        action="store_true",
        help=(
            "For annotation frames just outside the BeGaze frame range, reuse the "
            "nearest available gaze sample. This guarantees fewer empty rows but "
            "extrapolates beyond observed gaze data."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def clip_video_id(path: Path) -> str:
    name = path.parent.name
    parts = name.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:3])
    return name


def find_gaze_files(gaze_dirs: list[Path]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for gaze_dir in gaze_dirs:
        if not gaze_dir.exists():
            continue
        for path in sorted(gaze_dir.glob("*.txt")):
            files.setdefault(path.stem, path)
    return files


def gaze_file_candidates(video_id: str) -> list[str]:
    candidates = [video_id]
    if video_id.startswith("OP"):
        candidates.append(video_id[1:])
    elif video_id.startswith("P"):
        candidates.append("O" + video_id)
    return candidates


def parse_frame(value: str, fps: float) -> int:
    value = str(value).strip()
    if not value:
        raise ValueError("empty frame")
    if ":" not in value:
        return int(float(value))

    parts = value.split(":")
    if len(parts) != 4:
        raise ValueError(f"unsupported frame timecode: {value}")
    hours, minutes, seconds, frame = (int(float(part)) for part in parts)
    return int(round(((hours * 3600) + (minutes * 60) + seconds) * fps + frame))


def safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value or value == "-":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def gaze_from_row(row: dict[str, str]) -> tuple[float, float] | None:
    bx = safe_float(row.get("B POR X [px]"))
    by = safe_float(row.get("B POR Y [px]"))
    if bx is not None and by is not None:
        return bx, by

    left = (safe_float(row.get("L POR X [px]")), safe_float(row.get("L POR Y [px]")))
    right = (safe_float(row.get("R POR X [px]")), safe_float(row.get("R POR Y [px]")))
    valid = [(x, y) for x, y in (left, right) if x is not None and y is not None]
    if not valid:
        return None
    return (
        sum(x for x, _ in valid) / len(valid),
        sum(y for _, y in valid) / len(valid),
    )


def interpolate_gaze(gaze_by_frame: dict[int, tuple[float, float]]) -> dict[int, tuple[float, float]]:
    if len(gaze_by_frame) < 2:
        return gaze_by_frame

    frames = sorted(gaze_by_frame)
    filled: dict[int, tuple[float, float]] = {}
    for left_frame, right_frame in zip(frames, frames[1:]):
        left_x, left_y = gaze_by_frame[left_frame]
        right_x, right_y = gaze_by_frame[right_frame]
        filled[left_frame] = (left_x, left_y)
        gap = right_frame - left_frame
        if gap <= 1:
            continue
        for frame in range(left_frame + 1, right_frame):
            alpha = (frame - left_frame) / gap
            filled[frame] = (
                left_x + alpha * (right_x - left_x),
                left_y + alpha * (right_y - left_y),
            )
    filled[frames[-1]] = gaze_by_frame[frames[-1]]
    return filled


def read_gaze_file(path: Path, fps: float, interpolate: bool) -> dict[int, tuple[float, float]]:
    by_frame: dict[int, list[tuple[float, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        header = None
        for line in f:
            if line.startswith("Time\tType\t"):
                header = line.rstrip("\n").split("\t")
                break
        if header is None:
            raise ValueError(f"No BeGaze table header found in {path}")

        reader = csv.DictReader(f, fieldnames=header, delimiter="\t")
        for row in reader:
            if row.get("Type") != "SMP":
                continue
            try:
                frame = parse_frame(row["Frame"], fps)
            except (KeyError, TypeError, ValueError):
                continue
            gaze = gaze_from_row(row)
            if gaze is None:
                continue
            by_frame[frame].append(gaze)

    gaze_by_frame = {
        frame: (
            sum(x for x, _ in coords) / len(coords),
            sum(y for _, y in coords) / len(coords),
        )
        for frame, coords in by_frame.items()
        if coords
    }
    if interpolate:
        gaze_by_frame = interpolate_gaze(gaze_by_frame)
    return gaze_by_frame


def annotation_paths(roots: list[Path], pattern: str) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(sorted(root.rglob(pattern)))
    return paths


def row_frame(row: dict[str, str]) -> int | None:
    frame_raw = row.get("egtea_frame_number") or row.get("source_frame_index") or row.get("frame_index")
    try:
        return int(float(frame_raw))
    except (TypeError, ValueError):
        return None


def annotation_frames(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [
            frame
            for frame in (row_frame(row) for row in csv.DictReader(f))
            if frame is not None
        ]


def choose_gaze(
    path: Path,
    video_id: str,
    gaze_files: dict[str, Path],
    cache: dict[str, dict[int, tuple[float, float]]],
    fps: float,
    interpolate: bool,
) -> tuple[Path | None, dict[int, tuple[float, float]] | None]:
    frames = annotation_frames(path)
    candidates = [
        gaze_files[candidate]
        for candidate in gaze_file_candidates(video_id)
        if candidate in gaze_files
    ]
    if not candidates:
        return None, None

    best_path = candidates[0]
    best_gaze: dict[int, tuple[float, float]] | None = None
    best_covered = -1
    for candidate_path in candidates:
        cache_key = candidate_path.stem
        if cache_key not in cache:
            cache[cache_key] = read_gaze_file(candidate_path, fps, interpolate)
        gaze = cache[cache_key]
        covered = sum(1 for frame in frames if frame in gaze)
        if covered > best_covered:
            best_path = candidate_path
            best_gaze = gaze
            best_covered = covered
        if covered == len(frames):
            break
    return best_path, best_gaze


def progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def edge_gaze(frame: int, gaze_by_frame: dict[int, tuple[float, float]]) -> tuple[float, float] | None:
    if not gaze_by_frame:
        return None
    min_frame = min(gaze_by_frame)
    max_frame = max(gaze_by_frame)
    if frame < min_frame:
        return gaze_by_frame[min_frame]
    if frame > max_frame:
        return gaze_by_frame[max_frame]
    return None


def update_annotation(
    path: Path,
    gaze_by_frame: dict[int, tuple[float, float]],
    dry_run: bool,
    edge_fill: bool,
) -> dict[str, int]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    for col in ("gaze_x", "gaze_y"):
        if col not in fieldnames:
            fieldnames.append(col)

    filled = 0
    missing = 0
    for row in rows:
        frame = row_frame(row)
        if frame is None:
            missing += 1
            continue
        gaze = gaze_by_frame.get(frame)
        if gaze is None:
            gaze = edge_gaze(frame, gaze_by_frame) if edge_fill else None
            if gaze is None:
                missing += 1
                continue
        row["gaze_x"] = f"{gaze[0]:.2f}"
        row["gaze_y"] = f"{gaze[1]:.2f}"
        filled += 1

    if not dry_run:
        tmp = path.with_name(f".{path.name}.tmp")
        with tmp.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(path)

    return {"rows": len(rows), "filled": filled, "missing": missing}


def main() -> int:
    args = parse_args()
    gaze_dirs = args.gaze_dir or DEFAULT_GAZE_DIRS
    roots = args.annotation_root or DEFAULT_ANNOTATION_ROOTS

    gaze_files = find_gaze_files(gaze_dirs)
    if not gaze_files:
        raise FileNotFoundError(f"No gaze .txt files found under: {gaze_dirs}")

    cache: dict[str, dict[int, tuple[float, float]]] = {}
    totals = {"files": 0, "rows": 0, "filled": 0, "missing": 0, "no_gaze_file": 0}
    paths = annotation_paths(roots, args.pattern)
    if args.limit:
        paths = paths[: args.limit]

    for path in progress(paths, desc="Processing annotations"):
        video_id = clip_video_id(path)
        gaze_file, gaze_by_frame = choose_gaze(
            path,
            video_id,
            gaze_files,
            cache,
            args.fps,
            not args.no_interpolate,
        )
        if gaze_file is None or gaze_by_frame is None:
            totals["no_gaze_file"] += 1
            continue
        stats = update_annotation(path, gaze_by_frame, args.dry_run, args.edge_fill)
        totals["files"] += 1
        totals["rows"] += stats["rows"]
        totals["filled"] += stats["filled"]
        totals["missing"] += stats["missing"]

    mode = "dry-run" if args.dry_run else "updated"
    print(
        f"{mode}: files={totals['files']} rows={totals['rows']} "
        f"filled={totals['filled']} missing={totals['missing']} "
        f"no_gaze_file={totals['no_gaze_file']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

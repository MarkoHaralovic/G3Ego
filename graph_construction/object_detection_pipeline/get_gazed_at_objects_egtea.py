#!/usr/bin/env python3
from __future__ import annotations
import argparse
import pickle
from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError

DEFAULT_INPUT_ROOT = Path(
    "/projects/eemcs/dmb/ComputerVision/ego_graphs/vlm_datasets/egtea_gaze/"
    "vlm_ann_Qwen3-VL-32B-Instruct"
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--parse-name", default="parse_annotation.csv")
    parser.add_argument("--grounding-name", default="grounding_results_gdino_base.pkl")
    return parser.parse_args()

def resolve_grounding_pickle_path(clip_path: Path, grounding_name: str) -> Path | None:
    candidates = [clip_path / grounding_name]
    candidates.extend(sorted(clip_path.glob("grounding_results_*.pkl")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None

def normalize_phrase(value) -> str:
    if value is None or pd.isna(value):
        return ""
    value = str(value).strip()
    if value.lower() in {"nan", "none", "null"}:
        return ""
    return value

def populate_gazed_at_from_pickle(input_root: Path, parse_name: str, grounding_name: str) -> None:
    parse_paths = sorted(input_root.rglob(parse_name))

    overall_total = 0
    overall_covered = 0
    processed = 0

    for parse_path in parse_paths:
        clip_path = parse_path.parent
        pickle_path = resolve_grounding_pickle_path(clip_path, grounding_name)
        if pickle_path is None:
            continue

        with pickle_path.open("rb") as f:
            object_dict = pickle.load(f)

        parse_annotations = pd.read_csv(parse_path)
        parse_annotations["gazed_at_object"] = ""

        for row_idx, row in parse_annotations.iterrows():
            frame_key = str(row.get("frame_file") or f"{int(row['frame_id']):06d}.jpg")
            if frame_key not in object_dict:
                continue
            gazed_info = object_dict[frame_key].get("object_gazed_at", {})
            phrase = normalize_phrase(gazed_info.get("phrase") if gazed_info else None)
            if phrase:
                parse_annotations.loc[row_idx, "gazed_at_object"] = phrase

        parse_annotations.to_csv(parse_path, index=False, na_rep="")
        clip_total = len(parse_annotations)
        clip_covered = (
            parse_annotations["gazed_at_object"].astype(str).str.strip() != ""
        ).sum()
        overall_total += clip_total
        overall_covered += clip_covered
        processed += 1

    coverage_pct = (100.0 * overall_covered / overall_total) if overall_total else 0.0
    print(
        f"processed_clips={processed} coverage={overall_covered}/{overall_total} "
        f"({coverage_pct:.2f}%)"
    )


def main() -> int:
    args = parse_args()
    populate_gazed_at_from_pickle(
        args.input_root,
        args.parse_name,
        args.grounding_name
    )
    return 0


if __name__ == "__main__":
    main()

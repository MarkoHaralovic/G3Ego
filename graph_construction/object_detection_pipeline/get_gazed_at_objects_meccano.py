import os
import pickle

import pandas as pd

INPUT_DATASET_PATH = "/path/to/vlm_datasets/MECCANO_vlm_ann_Qwen3-VL-32B-Instruct-3fps"
SPLITS = ("Train", "Val", "Test")


def resolve_grounding_pickle_path(clip_path):
    candidates = [
        "grounding_results_gdino_base.pkl",
        "object_features_dinoT.pkl",
    ]
    for file_name in candidates:
        pickle_path = os.path.join(clip_path, file_name)
        if os.path.exists(pickle_path):
            return pickle_path

    for file_name in sorted(os.listdir(clip_path)):
        if file_name.startswith("grounding_results_") and file_name.endswith(".pkl"):
            return os.path.join(clip_path, file_name)

    return None


def frame_file_from_parse_row(parse_row):
    frame_id = int(parse_row["frame_id"])
    # MECCANO parse rows correspond to every 3rd RGB frame starting from 1.
    return f"{frame_id * 3 + 1:05d}.jpg"


def populate_gazed_at_from_pickle(input_dataset_path):
    overall_total = 0
    overall_covered = 0

    for split in SPLITS:
        split_path = os.path.join(input_dataset_path, split)
        if not os.path.isdir(split_path):
            print(f"Split not found, skipping: {split_path}")
            continue

        clip_names = [
            clip
            for clip in os.listdir(split_path)
            if os.path.isdir(os.path.join(split_path, clip))
        ]

        split_total = 0
        split_covered = 0
        processed_clips = 0

        for clip_name in clip_names:
            clip_path = os.path.join(split_path, clip_name)
            pickle_path = resolve_grounding_pickle_path(clip_path)
            csv_path = os.path.join(clip_path, "parse_annotation.csv")

            if pickle_path is None:
                print(f"Skipping {split}/{clip_name}: pickle not found")
                continue
            if not os.path.exists(csv_path):
                print(f"Skipping {split}/{clip_name}: parse_annotation.csv not found")
                continue

            with open(pickle_path, "rb") as f:
                object_dict = pickle.load(f)

            parse_annotations = pd.read_csv(csv_path)
            processed_clips += 1

            if "gazed_at_object" not in parse_annotations.columns:
                parse_annotations["gazed_at_object"] = None
            else:
                parse_annotations["gazed_at_object"] = None

            for frame_idx in range(len(parse_annotations)):
                frame_key = frame_file_from_parse_row(parse_annotations.iloc[frame_idx])
                if frame_key in object_dict:
                    gazed_info = object_dict[frame_key].get("object_gazed_at", {})
                    phrase = gazed_info.get("phrase") if gazed_info else None
                    if phrase:
                        parse_annotations.loc[frame_idx, "gazed_at_object"] = gazed_info[
                            "phrase"
                        ]

            parse_annotations.to_csv(csv_path, index=False)

            clip_total = len(parse_annotations)
            clip_covered = parse_annotations["gazed_at_object"].notna().sum()
            split_total += clip_total
            split_covered += clip_covered

        overall_total += split_total
        overall_covered += split_covered
        coverage_pct = (100.0 * split_covered / split_total) if split_total else 0.0
        print(
            f"{split}: processed {processed_clips} clips | "
            f"coverage {split_covered}/{split_total} ({coverage_pct:.2f}%)"
        )

    overall_pct = (100.0 * overall_covered / overall_total) if overall_total else 0.0
    print(
        f"Overall coverage: {overall_covered}/{overall_total} ({overall_pct:.2f}%)"
    )


if __name__ == "__main__":
    populate_gazed_at_from_pickle(INPUT_DATASET_PATH)

import os
import pickle
from pathlib import Path

import pandas as pd

INPUT_DATASET_PATH = "/home/s3758869/vlm_datasets/MECCANO_vlm_ann_Qwen3-VL-32B-Instruct-3fps"
SPLITS = ("Train", "Val", "Test")


def populate_gazed_at_from_pickle(input_dataset_path):
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

        for clip_name in clip_names:
            pickle_path = os.path.join(
                split_path, clip_name, "object_features_dinoT.pkl"
            )
            csv_path = os.path.join(split_path, clip_name, "parse_annotation.csv")

            if not os.path.exists(pickle_path):
                print(f"Skipping {split}/{clip_name}: pickle not found")
                continue
            if not os.path.exists(csv_path):
                print(f"Skipping {split}/{clip_name}: parse_annotation.csv not found")
                continue

            with open(pickle_path, "rb") as f:
                object_dict = pickle.load(f)

            parse_annotations = pd.read_csv(csv_path)

            if "gazed_at_object" not in parse_annotations.columns:
                parse_annotations["gazed_at_object"] = None

            for frame_idx in range(len(parse_annotations)):
                frame_key = f"frame_{frame_idx}"
                if frame_key in object_dict:
                    gazed_info = object_dict[frame_key].get("object_gazed_at", {})
                    if gazed_info and "phrase" in gazed_info:
                        parse_annotations.loc[frame_idx, "gazed_at_object"] = gazed_info[
                            "phrase"
                        ]

            parse_annotations.to_csv(csv_path, index=False)


if __name__ == "__main__":
    populate_gazed_at_from_pickle(INPUT_DATASET_PATH)

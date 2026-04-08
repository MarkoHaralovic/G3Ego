import os
import pickle
from pathlib import Path

import pandas as pd

INPUT_DATASET_PATH = "/home/s3758869/vlm_datasets/AriaEA_vlm_ann_3_10_llava-v1.6-34b-hf"


def populate_gazed_at_from_pickle(input_dataset_path):

    clip_names = [
        clip
        for clip in os.listdir(input_dataset_path)
        if os.path.isdir(os.path.join(input_dataset_path, clip))
    ]

    for clip_name in clip_names:
        pickle_path = os.path.join(
            input_dataset_path, clip_name, "object_features_dinoT.pkl"
        )
        csv_path = os.path.join(input_dataset_path, clip_name, "parse_annotation.csv")
        output_csv_path = os.path.join(
            input_dataset_path, clip_name, "parse_annotation.csv"
        )

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

        parse_annotations.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    populate_gazed_at_from_pickle(INPUT_DATASET_PATH)

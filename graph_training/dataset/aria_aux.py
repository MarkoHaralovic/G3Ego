import json
import os
import pickle
from ast import literal_eval

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset

from global_feature_training.data_loading.dataset_split_aria import (
    decode_label as decode_aria_label,
    map_or_skip_label as map_or_skip_aria_label,
    stratified_split as stratified_split_aria,
)

TRAIN_SIZE = 0.8
VAL_SIZE = 0.2

ignored_verbs = ["toast", "fold", "take", "put", "cut", "play"]
ignored_nouns = [
    "sink",
    "washer",
    "headphones",
    "camera",
    "watch",
    "pan",
    "tv",
    "coffee maker",
    "table",
    "book",
    "washer",
    "sink",
    "cupboard",
    "pants",
    "present",
    "macaroni",
    "pizza",
    "snack",
    "banana",
    "toast",
    "bag",
    "pancake",
    "chip",
    "newspaper",
    "magazine",
    "soup",
    "toast",
    "treadmill",
    "bag",
    "kitchen",
    "inside",
    "house",
    "door",
    "floor",
    "clothes",
    "pan",
    "cup",
    "fruit",
    "ceiling",
    "paper",
    "note",
    "room",
    "carpet",
    "table",
    "floor",
]
noun_replacement = "other"
skip_labels = {"na", "not_annotated"}

DATASET_PATH = "/path/to/vlm_datasets/AriaEA_vlm_ann_3_10_llava-v1.6-34b-hf"
model_name = "dinov3h16+"
pooling = "concat"

clips = [
    clip
    for clip in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, clip))
]

def return_train_val_samples(
    input_folder=DATASET_PATH,
    clips=clips,
    model_name=model_name,
    num_frames=None,
    pooling=pooling,
    skip_labels=skip_labels,
    skip_verbs=ignored_verbs,
    skip_nouns=ignored_nouns,
    noun_replacement="other",
    skip_na=True,
    val_ratio=VAL_SIZE,
):
    samples = collect_samples(
        input_folder=input_folder,
        clip_names=clips,
        model_name=model_name,
        pooling=pooling,
        num_frames=num_frames,
        skip_labels=skip_labels,
        skip_verbs=skip_verbs,
        skip_nouns=skip_nouns,
        noun_replacement=noun_replacement,
        skip_na=skip_na,
    )
    train_samples, val_samples = stratified_split_aria(samples, val_ratio, seed=0)

    acts = sorted({s[3] for s in samples})
    activity_to_idx = {a: i for i, a in enumerate(acts)}

    return train_samples, val_samples, activity_to_idx


def collect_samples(
    input_folder,
    clip_names,
    model_name,
    pooling=None,
    num_frames=None,
    skip_labels=set(),
    skip_verbs=set(),
    skip_nouns=set(),
    skip_activities=["no_annotated", "na"],
    noun_replacement="other",
    skip_na=True,
):

    samples = []  # (clip_name, h5_path, block_idx, label_str)

    for clip_name in clip_names:
        clip_path = os.path.join(input_folder, clip_name)

        frames_path = os.path.join(clip_path, "frames")
        annotations = pd.read_csv(os.path.join(clip_path, "annotations.csv"))
        parse_annotations = pd.read_csv(os.path.join(clip_path, "parse_annotation.csv"))
        object_features_path = os.path.join(clip_path, "object_features_dinoT.pkl")
        with open(object_features_path, "rb") as f:
            object_features = pickle.load(f)

        if pooling is not None:
            h5_path = os.path.join(
                clip_path, f"activity_features_model_{model_name}_pooling_{pooling}.h5"
            )
        elif num_frames is not None:
            h5_path = os.path.join(
                clip_path,
                f"activity_features_model_{model_name}_numframes_{num_frames}.h5",
            )
        else:
            raise Exception(f"Define either num_frames or pooling.")

        if not os.path.exists(h5_path):
            continue

        with h5py.File(h5_path, "r") as f:
            labels = f["activity_labels"][:]
            for block_idx, raw_label in enumerate(labels):
                if raw_label is skip_activities:
                    continue
                frame_anns = annotations[annotations["activity_block_id"] == block_idx]
                frame_idxs = list(
                    frame_anns["frame_index"]
                )  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                frame_clips = f["visual_features"][block_idx]
                frame_clips = frame_clips.reshape(10, 1280)
                frame_parse_anns_list = [
                    parse_annotations[parse_annotations["frame_id"] == f_idx]
                    for f_idx in frame_idxs
                ]
                frame_parse_anns = (
                    pd.concat(frame_parse_anns_list, ignore_index=False)
                    if frame_parse_anns_list
                    else pd.DataFrame()
                )
                frame_object_features = [
                    object_features[f"frame_{int(ind)}"] for ind in frame_idxs
                ]
                lab = decode_aria_label(raw_label)
                lab = map_or_skip_aria_label(
                    lab, skip_labels, skip_verbs, skip_nouns, noun_replacement, skip_na
                )
                if lab is not None:
                    samples.append(
                        (
                            clip_name,
                            h5_path,
                            block_idx,
                            lab,
                            frame_anns,
                            frame_parse_anns,
                            frame_clips,
                            frame_object_features,
                        )
                    )

    return samples

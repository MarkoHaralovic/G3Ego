import json
import os
import random
from collections import defaultdict

import h5py

TRAIN_SIZE = 0.8
VAL_SIZE = 0.2

ignored_verbs = ["toast", "fold", "take", "put", "cut", "play"]
ignored_nouns = [
    "sink","washer","headphones","camera","watch","pan","tv","coffee maker","table","book","washer","sink","cupboard","pants","present","macaroni","pizza","snack","banana","toast","bag","pancake","chip","newspaper","magazine","soup","toast","treadmill","bag","kitchen","inside","house","door","floor","clothes","pan","cup","fruit","ceiling","paper","note","room","carpet","table","floor",
]
noun_replacement = "other"
skip_labels = {"na"}

DATASET_PATH = "/home/s3758869/vlm_datasets/AriaEA_vlm_ann_3_10_llava-v1.6-34b-hf"
model_name = "dinov3h16+"
pooling = "concat"

OUTPUT_PATH = f"/home/s3758869/vlm_datasets/AriaEA_vlm_ann_3_10_llava-v1.6-34b-hf/training_data_split_train_{TRAIN_SIZE}_val_{VAL_SIZE}.json"
clips = [
    clip
    for clip in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, clip))
]

def map_or_skip_label(
    label: str,
    skip_labels=set(),
    skip_verbs=set(),
    replace_nouns=set(),
    noun_replacement="other",
    skip_na=True,
):
    lab = label.strip()
    if skip_na and lab.lower() in {"na", "n/a", "none", ""}:
        return None

    if lab in skip_labels:
        return None
    if len(lab.split("_")) != 2:
        return None

    verb, noun = lab.split("_", 1)

    if verb in skip_verbs:
        return None

    if noun in replace_nouns:
        noun = noun_replacement

    return f"{verb}_{noun}"


def decode_label(x):
    return x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)

def collect_samples(
    input_folder,
    clip_names,
    model_name,
    pooling=None,
    num_frames=None,
    skip_labels=set(),
    skip_verbs=set(),
    skip_nouns=set(),
    noun_replacement="other",
    skip_na=True,
):
    samples = []  # (clip_name, h5_path, block_idx, label_str)

    for clip_name in clip_names:
        clip_path = os.path.join(input_folder, clip_name)
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
                lab = decode_label(raw_label)
                lab = map_or_skip_label(
                    lab, skip_labels, skip_verbs, skip_nouns, noun_replacement, skip_na
                )
                if lab is not None:
                    samples.append((clip_name, h5_path, block_idx, lab))

    return samples


def stratified_split(samples, val_ratio=0.2, seed=0, min_val_per_class=1):
    rng = random.Random(seed)
    by_label = defaultdict(list)

    for s in samples:
        by_label[s[3]].append(s)

    train, val = [], []
    for _, items in by_label.items():
        rng.shuffle(items)

        n = len(items)
        v = int(round(n * val_ratio))
        if n >= 2:
            v = max(min_val_per_class, v)
            v = min(v, n - 1)  # min 1 in train
        else:
            v = 0

        val.extend(items[:v])
        train.extend(items[v:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def return_train_val_samples(
    input_folder=DATASET_PATH,
    clips=clips,
    model_name=model_name,
    num_frames=None,
    pooling=None,
    skip_labels=skip_labels,
    skip_verbs=ignored_verbs,
    skip_nouns=ignored_nouns,
    noun_replacement="other",
    skip_na=True,
    val_ratio=VAL_SIZE,
):
    samples = collect_samples(
        input_folder,
        clips,
        model_name,
        pooling,
        num_frames,
        skip_labels,
        skip_verbs,
        skip_nouns,
        noun_replacement,
        skip_na,
    )
    train_samples, val_samples = stratified_split(samples, val_ratio, seed=0)

    acts = sorted({s[3] for s in samples})
    activity_to_idx = {a: i for i, a in enumerate(acts)}

    return train_samples, val_samples, activity_to_idx

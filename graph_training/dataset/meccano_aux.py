import csv
import hashlib
import os
import random

def get_split_name(split):
    stem = os.path.basename(str(split)).lower()
    if "train" in stem:
        return "Train"
    if "val" in stem:
        return "Val"
    if "test" in stem:
        return "Test"
    raise ValueError(f"Unsupported MECCANO split hint: {split}")


def resolve_meccano_split_root(metadata_root, split_name):
    split_name = get_split_name(split_name)
    split_root = os.path.join(metadata_root, split_name)
    if os.path.exists(os.path.join(split_root, "verbs.json")):
        return split_root

    if (
        os.path.basename(os.path.normpath(metadata_root)).lower() == split_name.lower()
        and os.path.exists(os.path.join(metadata_root, "verbs.json"))
    ):
        return metadata_root

    raise FileNotFoundError(
        f"Could not find MECCANO metadata for split '{split_name}' under {metadata_root}"
    )


def resolve_meccano_global_root(metadata_root):
    candidates = [
        metadata_root,
        os.path.dirname(os.path.normpath(metadata_root)),
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "global_objects.json")):
            return candidate
    return None

def get_available_frame_numbers(clip_dir):
    annotations_path = os.path.join(clip_dir, "annotations_qwen3vl_32b_instruct.csv")
    if os.path.exists(annotations_path):
        with open(annotations_path, "r") as f:
            reader = csv.DictReader(f)
            return [int(row["frame_file"].split(".")[0]) for row in reader]

    parse_annotation_path = os.path.join(clip_dir, "parse_annotation.csv")
    with open(parse_annotation_path, "r") as f:
        reader = csv.DictReader(f)
        return [int(row["frame_id"]) * 3 + 1 for row in reader]

def select_action_frame_numbers(frame_numbers, num_graphs, selection_parts):
    if not frame_numbers:
        return None

    def _stable_seed(*selected_parts):
        selection_key = "::".join(str(part) for part in selected_parts)
        return int(hashlib.md5(selection_key.encode("utf-8")).hexdigest()[:8], 16)

    frame_numbers = sorted(frame_numbers)
    if len(frame_numbers) >= num_graphs:
        rng = random.Random(_stable_seed(*selection_parts))
        frame_numbers = sorted(rng.sample(frame_numbers, num_graphs))
    else:
        frame_numbers = frame_numbers + [frame_numbers[-1]] * (num_graphs - len(frame_numbers))

    return frame_numbers


def build_meccano_action_mapping(*actions_csv_paths):
    activity_to_idx = {}

    for actions_csv_path in actions_csv_paths:
        if actions_csv_path is None:
            continue

        with open(actions_csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                action_name = row["action_name"]
                action_id = int(row["action_id"])

                if action_name in activity_to_idx and activity_to_idx[action_name] != action_id:
                    raise ValueError(
                        f"Inconsistent mapping for {action_name}: "
                        f"{activity_to_idx[action_name]} vs {action_id}"
                    )

                activity_to_idx[action_name] = action_id

    if not activity_to_idx:
        raise ValueError("No MECCANO actions were found to build the class mapping.")

    return dict(sorted(activity_to_idx.items(), key=lambda item: item[1]))


def collect_meccano_samples(dataset_root, actions_csv_path, num_graphs=10, split_name=None):
    split_name = split_name or get_split_name(actions_csv_path)
    samples = []
    skipped_missing_clip = 0
    skipped_zero_frames = 0
    clip_frame_cache = {}

    with open(actions_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for sample_id, row in enumerate(reader):
            clip_name = row["video_id"]
            clip_dir = os.path.join(dataset_root, split_name, clip_name)
            if not os.path.isdir(clip_dir):
                skipped_missing_clip += 1
                continue

            available_frame_numbers = clip_frame_cache.get(clip_name)
            if available_frame_numbers is None:
                available_frame_numbers = get_available_frame_numbers(clip_dir)
                clip_frame_cache[clip_name] = available_frame_numbers

            start_frame = int(row["start_frame"].split(".")[0])
            end_frame = int(row["end_frame"].split(".")[0])
            in_span = [
                frame_number
                for frame_number in available_frame_numbers
                if start_frame <= frame_number <= end_frame
            ]
            selected = select_action_frame_numbers(
                in_span,
                num_graphs=num_graphs,
                selection_parts=(
                    split_name,
                    clip_name,
                    sample_id,
                    row["action_name"],
                    start_frame,
                    end_frame,
                ),
            )
            if selected is None:
                skipped_zero_frames += 1
                continue

            samples.append(
                {
                    "clip_name": clip_name,
                    "clip_dir": clip_dir,
                    "sample_id": sample_id,
                    "label": row["action_name"],
                    "frame_numbers": selected,
                    "action_id": int(row["action_id"]),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "split_name": split_name,
                }
            )

    stats = {
        "split_name": split_name,
        "num_samples": len(samples),
        "skipped_missing_clip": skipped_missing_clip,
        "skipped_zero_frames": skipped_zero_frames,
    }
    return samples, stats


def return_meccano_train_test_samples(
    dataset_root,
    train_actions_csv,
    test_actions_csv,
    num_graphs=10,
):
    train_samples, train_stats = collect_meccano_samples(
        dataset_root=dataset_root,
        actions_csv_path=train_actions_csv,
        num_graphs=num_graphs,
        split_name="Train",
    )
    test_samples, test_stats = collect_meccano_samples(
        dataset_root=dataset_root,
        actions_csv_path=test_actions_csv,
        num_graphs=num_graphs,
        split_name="Test",
    )

    activity_to_idx = build_meccano_action_mapping(
        train_actions_csv,
        test_actions_csv,
    )

    return train_samples, test_samples, activity_to_idx, {
        "train": train_stats,
        "test": test_stats,
    }


def return_meccano_train_val_test_samples(
    dataset_root,
    train_actions_csv,
    val_actions_csv,
    test_actions_csv,
    num_graphs=10,
):
    train_samples, train_stats = collect_meccano_samples(
        dataset_root=dataset_root,
        actions_csv_path=train_actions_csv,
        num_graphs=num_graphs,
        split_name="Train",
    )
    val_samples, val_stats = collect_meccano_samples(
        dataset_root=dataset_root,
        actions_csv_path=val_actions_csv,
        num_graphs=num_graphs,
        split_name="Val",
    )
    test_samples, test_stats = collect_meccano_samples(
        dataset_root=dataset_root,
        actions_csv_path=test_actions_csv,
        num_graphs=num_graphs,
        split_name="Test",
    )

    activity_to_idx = build_meccano_action_mapping(
        train_actions_csv,
        val_actions_csv,
        test_actions_csv,
    )

    return train_samples, val_samples, test_samples, activity_to_idx, {
        "train": train_stats,
        "val": val_stats,
        "test": test_stats,
    }


def load_clip_text_embeddings(clip_text_path, emb_dim=512):
    zero_factory = partial(torch.zeros, emb_dim, dtype=torch.float32)
    if not os.path.exists(clip_text_path):
        return defaultdict(zero_factory)
    with open(clip_text_path, "rb") as f:
        return defaultdict(zero_factory, pickle.load(f))
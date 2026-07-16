from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from ast import literal_eval
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

GRAPH_TRAINING_ROOT = Path(__file__).resolve().parent
REPO_ROOT = GRAPH_TRAINING_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(GRAPH_TRAINING_ROOT))

from dataset.GraphDataset import GraphDatasetEgtea
from graph_construction.graphs.full_graph import to_singular
from train.evaluate import (
    build_loss_fn,
    compute_class_weights,
    store_model,
)
from train.train import do_epoch
from train.utils import (
    build_graph_mlp,
    build_optimizer,
    build_scheduler,
    evaluate_checkpoint,
    graph_relation_count,
    make_loaders,
    resolve_device,
    save_json,
    write_training_outputs,
)


def parse_jsonish(value, default):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, (dict, list)):
        return value
    value = str(value).strip()
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return literal_eval(value)


def clip_video_id(clip_name):
    return "-".join(str(clip_name).split("-")[:3])


def select_frame_numbers(frame_numbers, num_graphs):
    frame_numbers = sorted(set(int(frame_number) for frame_number in frame_numbers))
    if not frame_numbers:
        return None
    if len(frame_numbers) <= num_graphs:
        return frame_numbers
    if num_graphs == 1:
        return [frame_numbers[len(frame_numbers) // 2]]
    indices = torch.linspace(0, len(frame_numbers) - 1, steps=num_graphs).round().long()
    return [frame_numbers[idx] for idx in indices.tolist()]


def read_split_action_map(split_file):
    action_map = {}
    if not split_file or not os.path.exists(split_file):
        return action_map
    with open(split_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                action_map[parts[0]] = int(parts[1]) - 1
    return action_map


def read_idx_labels(path):
    labels = {}
    if not path or not os.path.exists(path):
        return labels
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label, _, idx = line.rpartition(" ")
            if idx.isdigit():
                labels[int(idx) - 1] = label.strip()
    return labels


def resolve_feature_dir(feature_root, clip_name):
    video_id = clip_video_id(clip_name)
    candidates = [
        os.path.join(feature_root, video_id, clip_name),
    ]
    if video_id.startswith("P"):
        candidates.append(os.path.join(feature_root, "O" + video_id, clip_name))
    elif video_id.startswith("OP"):
        candidates.append(os.path.join(feature_root, video_id[1:], clip_name))

    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return None


def collect_split_samples(
    split_root,
    feature_root,
    num_graphs,
    rgb_feature_filename,
    split_actions_path=None,
    action_idx_path=None,
):
    split_actions = read_split_action_map(split_actions_path)
    action_labels = read_idx_labels(action_idx_path)
    samples = []
    skipped_missing_feature_dir = 0
    skipped_missing_feature_file = 0
    skipped_zero_frames = 0

    clip_dirs = sorted(
        str(path)
        for path in Path(split_root).iterdir()
        if path.is_dir() and (path / "parse_annotation.csv").exists()
    )
    for sample_id, clip_dir in enumerate(clip_dirs):
        clip_name = os.path.basename(clip_dir)
        feature_dir = resolve_feature_dir(feature_root, clip_name)
        if feature_dir is None:
            skipped_missing_feature_dir += 1
            continue
        if not os.path.exists(os.path.join(feature_dir, rgb_feature_filename)):
            skipped_missing_feature_file += 1
            continue

        parse_annotations = pd.read_csv(os.path.join(clip_dir, "parse_annotation.csv"))
        frame_numbers = [
            int(Path(str(frame_file)).stem)
            for frame_file in parse_annotations["frame_file"].dropna().tolist()
        ]
        selected = select_frame_numbers(frame_numbers, num_graphs)
        if selected is None:
            skipped_zero_frames += 1
            continue

        first_row = parse_annotations.iloc[0]
        action_id = first_row.get("action_id")
        if pd.isna(action_id) or str(action_id) == "":
            action_id = split_actions.get(clip_name)
        else:
            action_id = int(action_id)
        label = str(first_row.get("source_action") or action_labels.get(action_id) or action_id)

        samples.append(
            {
                "clip_name": clip_name,
                "clip_dir": clip_dir,
                "feature_dir": feature_dir,
                "sample_id": sample_id,
                "label": label,
                "action_id": int(action_id),
                "frame_numbers": selected,
            }
        )

    stats = {
        "split_root": str(split_root),
        "num_samples": len(samples),
        "skipped_missing_feature_dir": skipped_missing_feature_dir,
        "skipped_missing_feature_file": skipped_missing_feature_file,
        "skipped_zero_frames": skipped_zero_frames,
    }
    return samples, stats


def _stable_sample_key(sample):
    return (
        hashlib.md5(sample["clip_name"].encode("utf-8")).hexdigest(),
        sample["sample_id"],
    )


def _allocate_stratified_val_counts(samples_by_action, val_fraction, val_num_samples):
    min_counts = {}
    max_counts = {}
    for action_id, action_samples in samples_by_action.items():
        min_counts[action_id] = 1
        max_counts[action_id] = (
            len(action_samples) - 1 if len(action_samples) > 1 else 1
        )

    if val_num_samples is None:
        val_counts = {}
        for action_id, action_samples in samples_by_action.items():
            val_count = int(round(len(action_samples) * val_fraction))
            val_count = max(min_counts[action_id], val_count)
            val_counts[action_id] = min(max_counts[action_id], val_count)
        return val_counts

    target = int(val_num_samples)
    min_total = sum(min_counts.values())
    max_total = sum(max_counts.values())
    if target < min_total:
        raise ValueError(
            f"val_num_samples={target} is too small for stratified validation: "
            f"need at least {min_total} samples for {len(samples_by_action)} actions."
        )
    if target > max_total:
        raise ValueError(
            f"val_num_samples={target} is too large: at most {max_total} samples "
            "can be held out while keeping train samples for each action."
        )

    val_counts = dict(min_counts)
    remaining = target - min_total
    total_samples = sum(len(samples) for samples in samples_by_action.values())
    quotas = {}

    for action_id, action_samples in samples_by_action.items():
        capacity = max_counts[action_id] - min_counts[action_id]
        quota = remaining * len(action_samples) / float(total_samples)
        extra = min(int(quota), capacity)
        val_counts[action_id] += extra
        quotas[action_id] = quota - extra

    current = sum(val_counts.values())
    while current < target:
        candidates = [
            action_id
            for action_id in samples_by_action
            if val_counts[action_id] < max_counts[action_id]
        ]
        action_id = max(
            candidates,
            key=lambda item: (
                quotas[item],
                len(samples_by_action[item]),
                -int(item),
            ),
        )
        val_counts[action_id] += 1
        quotas[action_id] = 0.0
        current += 1

    while current > target:
        candidates = [
            action_id
            for action_id in samples_by_action
            if val_counts[action_id] > min_counts[action_id]
        ]
        action_id = min(
            candidates,
            key=lambda item: (
                quotas[item],
                len(samples_by_action[item]),
                int(item),
            ),
        )
        val_counts[action_id] -= 1
        current -= 1

    return val_counts


def _target_val_size(samples_by_action, val_fraction, val_num_samples):
    if val_num_samples is not None:
        return int(val_num_samples)
    return int(round(sum(len(samples) for samples in samples_by_action.values()) * val_fraction))


def _allocate_reference_val_counts(
    samples_by_action,
    reference_samples,
    val_fraction,
    val_num_samples,
):
    target = _target_val_size(samples_by_action, val_fraction, val_num_samples)
    if target <= 0:
        return {action_id: 0 for action_id in samples_by_action}

    capacities = {
        action_id: (len(action_samples) - 1 if len(action_samples) > 1 else 1)
        for action_id, action_samples in samples_by_action.items()
    }
    max_total = sum(capacities.values())
    if target > max_total:
        raise ValueError(
            f"Requested {target} validation samples, but at most {max_total} can "
            "be held out while preserving the available train split."
        )

    reference_counts = Counter(sample["action_id"] for sample in reference_samples)
    reference_total = sum(
        count for action_id, count in reference_counts.items()
        if action_id in samples_by_action
    )
    if reference_total <= 0:
        return _allocate_stratified_val_counts(
            samples_by_action, val_fraction, val_num_samples
        )

    positive_actions = [
        action_id
        for action_id, count in reference_counts.items()
        if count > 0 and action_id in samples_by_action and capacities[action_id] > 0
    ]
    min_counts = {action_id: 0 for action_id in samples_by_action}
    if target >= len(positive_actions):
        for action_id in positive_actions:
            min_counts[action_id] = 1

    val_counts = dict(min_counts)
    current = sum(val_counts.values())
    remaining = target - current
    quotas = {}

    for action_id in samples_by_action:
        ref_count = reference_counts.get(action_id, 0)
        quota = target * ref_count / float(reference_total)
        desired_extra = max(0.0, quota - val_counts[action_id])
        capacity = capacities[action_id] - val_counts[action_id]
        extra = min(int(desired_extra), capacity)
        val_counts[action_id] += extra
        quotas[action_id] = desired_extra - extra
        remaining -= extra

    while remaining > 0:
        candidates = [
            action_id
            for action_id in samples_by_action
            if val_counts[action_id] < capacities[action_id]
        ]
        if not candidates:
            break
        action_id = max(
            candidates,
            key=lambda item: (
                quotas.get(item, 0.0),
                reference_counts.get(item, 0),
                len(samples_by_action[item]),
                -int(item),
            ),
        )
        val_counts[action_id] += 1
        quotas[action_id] = 0.0
        remaining -= 1

    return val_counts


def split_train_val(
    samples,
    val_fraction,
    val_num_samples=None,
    reference_samples=None,
):
    if val_fraction <= 0 and val_num_samples is None:
        return samples, []

    samples_by_action = {}
    for sample in samples:
        samples_by_action.setdefault(sample["action_id"], []).append(sample)

    if reference_samples is not None:
        val_counts = _allocate_reference_val_counts(
            samples_by_action,
            reference_samples,
            val_fraction,
            val_num_samples,
        )
    else:
        val_counts = _allocate_stratified_val_counts(
            samples_by_action, val_fraction, val_num_samples
        )
    train_samples = []
    val_samples = []

    for action_id in sorted(samples_by_action):
        sorted_action_samples = sorted(samples_by_action[action_id], key=_stable_sample_key)
        val_count = val_counts[action_id]

        val_samples.extend(sorted_action_samples[:val_count])
        train_samples.extend(sorted_action_samples[val_count:])

    if samples and not val_samples:
        val_samples = [samples[0]]
        train_samples = samples[1:]

    sample_order = {id(sample): idx for idx, sample in enumerate(samples)}
    train_samples.sort(key=lambda sample: sample_order[id(sample)])
    val_samples.sort(key=lambda sample: sample_order[id(sample)])
    return train_samples, val_samples


def build_activity_mapping(samples):
    by_original_id = {}
    for sample in samples:
        by_original_id.setdefault(sample["action_id"], sample["label"])
    return {
        label: compact_idx
        for compact_idx, (_original_id, label) in enumerate(sorted(by_original_id.items()))
    }


def build_vocab(samples, metadata_root):
    verbs = Counter()
    relationships = Counter({"direct_object": 0, "aux_direct_object": 0, "aux_verb": 0})
    attrs = Counter()

    objects_path = os.path.join(metadata_root, "objects.json")
    with open(objects_path, "r") as f:
        objects = json.load(f)

    for sample in samples:
        rows = pd.read_csv(os.path.join(sample["clip_dir"], "parse_annotation.csv"))
        for row in rows.to_dict("records"):
            verb = row.get("verb")
            if verb and not pd.isna(verb):
                verbs[str(verb)] += 1

            all_objects = parse_jsonish(row.get("all_objects"), {})
            if not isinstance(all_objects, dict):
                all_objects = {}
            for obj_info in all_objects.values():
                for attr in obj_info.get("attributes", []):
                    attrs[str(attr)] += 1

            aux_verbs = parse_jsonish(row.get("aux_verbs"), [])
            for aux_verb in aux_verbs:
                verbs[str(aux_verb)] += 1
                relationships["aux_verb"] += 1

            aux_objects = parse_jsonish(row.get("object_aux_verb"), {})
            if isinstance(aux_objects, dict):
                relationships["aux_direct_object"] += sum(
                    len(items) for items in aux_objects.values()
                )

            rel_pairs = parse_jsonish(row.get("preposition_object_pairs"), [])
            for rel_pair in rel_pairs:
                for _obj_name, rel in rel_pair.items():
                    relationships[str(rel)] += 1

    return {
        "verbs": {verb: idx for idx, verb in enumerate(sorted(verbs.keys()))},
        "objects": objects,
        "relationships": {
            rel: idx for idx, rel in enumerate(sorted(relationships.keys()))
        },
        "attributes": {attr: idx for idx, attr in enumerate(sorted(attrs.keys()))},
    }


def build_train_sampler(dataset, activity_to_idx, sampler_cfg):
    if not sampler_cfg or sampler_cfg.get("name") in {None, "none"}:
        return None
    if sampler_cfg.get("name") != "balanced":
        raise ValueError(f"Unsupported sampler: {sampler_cfg.get('name')}")
    labels = [activity_to_idx[label_str] for _, _, label_str, *_ in dataset.sample_index]
    counts = Counter(labels)
    power = float(sampler_cfg.get("power", 1.0))
    weights = torch.tensor(
        [1.0 / (counts[label] ** power) for label in labels],
        dtype=torch.double,
    )
    return WeightedRandomSampler(
        weights,
        num_samples=int(sampler_cfg.get("num_samples", len(weights))),
        replacement=bool(sampler_cfg.get("replacement", True)),
    )


def apply_cli_overrides(args, config):
    if args.split_id is not None:
        config["data"]["split_id"] = int(args.split_id)
    if args.num_graphs is not None:
        config["mlp"]["num_graphs"] = int(args.num_graphs)
    if args.num_epochs is not None:
        config["training"]["num_epochs"] = int(args.num_epochs)
    if args.graph_type is not None:
        config["data"]["graph_type"] = args.graph_type
    if args.batch_size is not None:
        config["data"]["batch_size"] = int(args.batch_size)
    if args.val_num_samples is not None:
        config["data"]["val_num_samples"] = int(args.val_num_samples)
    if getattr(args, "val_distribution_source", None) is not None:
        config["data"]["val_distribution_source"] = args.val_distribution_source
    if args.experiment_name is not None:
        config["experiment_name"] = args.experiment_name
    if args.output_base_path is not None:
        config["output"]["base_path"] = args.output_base_path
    if args.easg_cache_path is not None:
        config["data"]["easg_cache_path"] = args.easg_cache_path


def main(args, config):
    apply_cli_overrides(args, config)
    mlp_cfg = config["mlp"]
    projector_cfg = mlp_cfg["projector"]
    attention_pool_cfg = mlp_cfg["attention_pooler"]
    data_cfg = config["data"]

    device = resolve_device(config["device"])

    num_graphs = mlp_cfg["num_graphs"]
    split_id = int(data_cfg.get("split_id", 1))
    input_root = data_cfg["input_path"]
    train_split_root = data_cfg.get(
        "train_split_root", os.path.join(input_root, "train", f"train_split_{split_id}")
    )
    test_split_root = data_cfg.get(
        "test_split_root", os.path.join(input_root, "test", f"test_split_{split_id}")
    )
    annotations_root = data_cfg.get(
        "annotations_root",
        "/path/to/ego_graphs/vlm_datasets/egtea_gaze/annotations",
    )
    train_actions_path = os.path.join(annotations_root, f"train_split{split_id}.txt")
    test_actions_path = os.path.join(annotations_root, f"test_split{split_id}.txt")
    action_idx_path = os.path.join(annotations_root, "action_idx.txt")

    train_all_samples, train_stats = collect_split_samples(
        train_split_root,
        data_cfg["feature_root"],
        num_graphs,
        data_cfg.get("rgb_feature_filename", "frame_features_model_dinov3_vitl16.h5"),
        split_actions_path=train_actions_path,
        action_idx_path=action_idx_path,
    )
    test_samples, test_stats = collect_split_samples(
        test_split_root,
        data_cfg["feature_root"],
        num_graphs,
        data_cfg.get("rgb_feature_filename", "frame_features_model_dinov3_vitl16.h5"),
        split_actions_path=test_actions_path,
        action_idx_path=action_idx_path,
    )
    val_distribution_source = str(
        data_cfg.get("val_distribution_source", "test")
    ).lower()
    val_reference_samples = test_samples if val_distribution_source == "test" else None
    train_samples, val_samples = split_train_val(
        train_all_samples,
        float(data_cfg.get("val_fraction", 0.1)),
        val_num_samples=data_cfg.get("val_num_samples"),
        reference_samples=val_reference_samples,
    )
    all_samples = train_samples + val_samples + test_samples
    activity_to_idx = build_activity_mapping(all_samples)
    vocab = build_vocab(all_samples, input_root)

    graph_type = data_cfg.get("graph_type", "full")
    num_rels = graph_relation_count(vocab["relationships"], graph_type)

    print(f"Running experiment: {config['experiment_name']}")
    print(f"EGTEA split id: {split_id}")
    print(f"len(train_samples): {len(train_samples)}")
    print(f"len(val_samples): {len(val_samples)}")
    print(f"len(test_samples): {len(test_samples)}")
    print(f"val_distribution_source: {val_distribution_source}")
    print(f"num_classes: {len(activity_to_idx)}")
    print(f"vocab sizes: verbs={len(vocab['verbs'])} objects={len(vocab['objects'])} rels={num_rels} attrs={len(vocab['attributes'])}")

    dataset_kwargs = {
        "activity_to_idx": activity_to_idx,
        "graph_type": graph_type,
        "vocab": vocab,
        "clip_text_path": data_cfg.get("clip_text_path"),
        "easg_cache_path": data_cfg.get("easg_cache_path"),
        "rgb_feature_filename": data_cfg.get(
            "rgb_feature_filename", "frame_features_model_dinov3_vitl16.h5"
        ),
    }
    train_dataset = GraphDatasetEgtea(train_samples, **dataset_kwargs)
    validation_dataset = GraphDatasetEgtea(val_samples, **dataset_kwargs)
    test_dataset = GraphDatasetEgtea(test_samples, **dataset_kwargs)

    sampler_cfg = config["training"].get("sampler", {"name": "none"})
    train_sampler = build_train_sampler(train_dataset, activity_to_idx, sampler_cfg)
    train_loader, val_loader, test_loader = make_loaders(
        (train_dataset, validation_dataset, test_dataset),
        data_cfg,
        train_sampler=train_sampler,
        eval_num_workers=0,
    )

    num_epochs = config["training"]["num_epochs"]
    model = build_graph_mlp(config, vocab, len(activity_to_idx), num_rels, device)

    class_weights = None
    if config["training"]["loss"]["ifw"]:
        class_weights = compute_class_weights(train_dataset, activity_to_idx)
        print(class_weights)
    loss_func = build_loss_fn(config["training"]["loss"], class_weights)

    opt = build_optimizer(model, config["training"])
    scheduler = build_scheduler(opt, config["training"])

    save_path = os.path.join(
        config["output"]["base_path"],
        config["experiment_name"],
        (
            f"dino_model_fc_layer_{mlp_cfg['fc_layers_num']}_num_epoch_{num_epochs}"
            f"_graph_emb_dim_{projector_cfg['graph_emb_dim']}"
            f"_final_graph_emb_dim_{attention_pool_cfg['final_graph_emb_dim']}"
            f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ),
    )
    os.makedirs(save_path, exist_ok=True)

    split_stats = {
        "train_all": train_stats,
        "test": test_stats,
        "train_after_val_split": len(train_samples),
        "val_after_val_split": len(val_samples),
    }
    save_json(os.path.join(save_path, "experiment_config.json"), {**config, "split_stats": split_stats})
    save_json(os.path.join(save_path, "class_mapping.json"), activity_to_idx)

    results = {}
    epoch_preds = {k: {} for k in range(num_epochs)}
    best_epoch_result = {"top1": -1, "top5": -1, "f1": -1}
    global_step = 0

    model.train()
    for epoch in tqdm(range(num_epochs), desc=f"Training for {num_epochs} epochs", unit="epoch"):
        print(f"Epoch {epoch + 1}/{num_epochs}\n")
        epoch_result, global_step, preds, targets = do_epoch(
            device=device,
            net=model,
            opt=opt,
            train_loader=train_loader,
            validate_loader=val_loader,
            global_step=global_step,
            num_classes_train=len(activity_to_idx),
            num_classes_val=len(activity_to_idx),
            loss_func=loss_func,
            train_progress_desc=f"Epoch {epoch + 1}/{num_epochs} train",
            val_progress_desc=f"Epoch {epoch + 1}/{num_epochs} val",
        )
        epoch_preds[epoch]["predictions"] = [
            train_dataset.idx_to_activity[i] for i in preds
        ]
        epoch_preds[epoch]["targets"] = [
            train_dataset.idx_to_activity[i] for i in targets
        ]
        val_metrics = epoch_result["val"]["eval_metrics"]
        print(
            "\nValidation metrics: "
            f"Top-1 {val_metrics['top1'] * 100:.2f}% | "
            f"Top-5 {val_metrics['top5'] * 100:.2f}% | "
            f"Avg. F1 {val_metrics['avg_f1'] * 100:.2f}%"
        )

        if val_metrics["top1"] > best_epoch_result["top1"]:
            best_epoch_result["top1"] = val_metrics["top1"]
            store_model(model, opt, epoch, save_path, metric="top1")
            store_model(model, opt, epoch, save_path, metric="acc")
        if val_metrics["top5"] > best_epoch_result["top5"]:
            best_epoch_result["top5"] = val_metrics["top5"]
            store_model(model, opt, epoch, save_path, metric="top5")
        if val_metrics["f1"] > best_epoch_result["f1"]:
            best_epoch_result["f1"] = val_metrics["f1"]
            store_model(model, opt, epoch, save_path, metric="f1")

        results[epoch] = epoch_result
        if scheduler is not None:
            scheduler.step()

    write_training_outputs(save_path, results, epoch_preds)

    top1_test_summary = evaluate_checkpoint(
        model, save_path, "top1", test_loader, test_dataset, device
    )
    top5_test_summary = evaluate_checkpoint(
        model, save_path, "top5", test_loader, test_dataset, device
    )
    save_json(os.path.join(save_path, "final_test_results_top1.json"), top1_test_summary)
    save_json(os.path.join(save_path, "final_test_results_top5.json"), top5_test_summary)
    save_json(os.path.join(save_path, "final_test_results.json"), top1_test_summary)

    print(f"Best validation Top-1 accuracy: {best_epoch_result['top1'] * 100:.2f}%")
    print(f"Best validation Top-5 accuracy: {best_epoch_result['top5'] * 100:.2f}%")
    print(f"Best validation F1: {best_epoch_result['f1'] * 100:.2f}%")
    print(
        "Best Top-1 checkpoint test metrics: "
        f"Top-1 {top1_test_summary['metrics']['top1'] * 100:.2f}% | "
        f"Top-5 {top1_test_summary['metrics']['top5'] * 100:.2f}% | "
        f"Avg. F1 {top1_test_summary['metrics']['avg_f1'] * 100:.2f}%"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to the experiment config JSON file"
    )
    parser.add_argument("--split-id", type=int, default=None)
    parser.add_argument("--num-graphs", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--graph-type", choices=("full", "pruned"), default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--val-num-samples", type=int, default=None)
    parser.add_argument(
        "--val-distribution-source",
        choices=("train", "test"),
        default=None,
    )
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--output-base-path", type=str, default=None)
    parser.add_argument("--easg-cache-path", type=str, default=None)
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        config = json.load(f)
    main(args, config)

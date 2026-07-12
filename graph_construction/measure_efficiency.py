from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import networkx as nx
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
graph_training_root = project_root / "graph_training"
for path in (project_root, graph_training_root):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from graph_construction.graphs.graph_to_nx import full_action_graph_to_nx
from graph_training.dataset.GraphDataset import GraphDatasetEgtea, GraphDatasetMeccano
from graph_training.dataset.meccano_aux import return_meccano_train_val_test_samples
from graph_training.train_graph_mlp_egtea import (
    build_activity_mapping,
    build_vocab,
    collect_split_samples,
    split_train_val,
)


DEFAULT_EGTEA_ROOT = Path(
    "/projects/eemcs/dmb/ComputerVision/ego_graphs/vlm_datasets/egtea_gaze"
)
DEFAULT_EGTEA_INPUT = DEFAULT_EGTEA_ROOT / "vlm_ann_Qwen3-VL-32B-Instruct"
DEFAULT_EGTEA_FEATURES = DEFAULT_EGTEA_ROOT / "framewise_videos/frame_cache_fps_30"
DEFAULT_EGTEA_ANNOTATIONS = DEFAULT_EGTEA_ROOT / "annotations"
DEFAULT_MECCANO_ROOT = Path(
    "/projects/eemcs/dmb/ComputerVision/ego_graphs/vlm_datasets/"
    "MECCANO_vlm_ann_Qwen3-VL-32B-Instruct-3fps"
)
MODEL_TO_GRAPH_TYPE = {"fasg": "full", "gpasg": "pruned"}


class MetricAccumulator:
    def __init__(self):
        self.graph_count = 0
        self.node_count = 0
        self.edge_count = 0
        self.max_shortest_path_sum = 0.0
        self.global_efficiency_sum = 0.0
        self.max_shortest_path = 0

    def update(self, graph):
        gnx, _ = full_action_graph_to_nx(graph, directed=True)
        graph_undir = gnx.to_undirected()
        per_graph_max = max_finite_shortest_path(graph_undir)
        global_eff = (
            nx.global_efficiency(graph_undir)
            if graph_undir.number_of_nodes() >= 2
            else 0.0
        )

        self.graph_count += 1
        self.node_count += graph_undir.number_of_nodes()
        self.edge_count += graph_undir.number_of_edges()
        self.max_shortest_path_sum += per_graph_max
        self.global_efficiency_sum += global_eff
        self.max_shortest_path = max(self.max_shortest_path, per_graph_max)

    def as_row(self, dataset, split, split_id, model_name, graph_type, sample_count):
        denom = max(self.graph_count, 1)
        return {
            "dataset": dataset,
            "split_id": split_id or "",
            "split": split,
            "model": model_name,
            "graph_type": graph_type,
            "samples": sample_count,
            "graphs": self.graph_count,
            "avg_nodes": self.node_count / denom,
            "avg_edges": self.edge_count / denom,
            "avg_max_shortest_path": self.max_shortest_path_sum / denom,
            "avg_global_efficiency": self.global_efficiency_sum / denom,
            "max_shortest_path": self.max_shortest_path,
        }


def max_finite_shortest_path(graph):
    if graph.number_of_nodes() < 2:
        return 0
    max_path = 0
    for _source, lengths in nx.all_pairs_shortest_path_length(graph):
        if lengths:
            max_path = max(max_path, max(lengths.values()))
    return max_path


def resolve_meccano_csvs(root, args):
    if args.meccano_train_csv and args.meccano_val_csv and args.meccano_test_csv:
        return args.meccano_train_csv, args.meccano_val_csv, args.meccano_test_csv

    candidates = [
        (
            root / "action_recognition/MECCANO_train_actions.csv",
            root / "action_recognition/MECCANO_val_actions.csv",
            root / "action_recognition/MECCANO_test_actions.csv",
        ),
        (
            root / "next_action_classification/MECCANO_train_next_actions_delta1s.csv",
            root / "next_action_classification/MECCANO_val_next_actions_delta1s.csv",
            root / "next_action_classification/MECCANO_test_next_actions_delta1s.csv",
        ),
    ]
    for paths in candidates:
        if all(path.exists() for path in paths):
            return tuple(str(path) for path in paths)
    raise FileNotFoundError(
        "Could not find MECCANO action CSVs. Pass --meccano-train-csv, "
        "--meccano-val-csv, and --meccano-test-csv."
    )


def measure_dataset(dataset, desc, max_samples=None):
    metrics = MetricAccumulator()
    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    for idx in tqdm(range(total), desc=desc, unit="sample"):
        item = dataset[idx]
        for graph in item["full_action_graphs"].values():
            metrics.update(graph)
    return metrics, total


def measure_datasets(datasets, desc, max_samples=None):
    metrics = MetricAccumulator()
    sample_count = 0
    for split_name, dataset in datasets.items():
        split_total = (
            len(dataset)
            if max_samples is None
            else min(len(dataset), max_samples)
        )
        sample_count += split_total
        for idx in tqdm(
            range(split_total),
            desc=f"{desc} {split_name}",
            unit="sample",
        ):
            item = dataset[idx]
            for graph in item["full_action_graphs"].values():
                metrics.update(graph)
    return metrics, sample_count


def build_meccano_datasets(args, graph_type):
    root = Path(args.meccano_root)
    train_csv, val_csv, test_csv = resolve_meccano_csvs(root, args)
    train_samples, val_samples, test_samples, activity_to_idx, _stats = (
        return_meccano_train_val_test_samples(
            str(root),
            train_csv,
            val_csv,
            test_csv,
            num_graphs=args.meccano_num_graphs,
        )
    )
    return {
        "Train": GraphDatasetMeccano(
            metadata_root=str(root),
            split_name="Train",
            samples=train_samples,
            activity_to_idx=activity_to_idx,
            graph_type=graph_type,
            easg_cache_path=None,
            rgb_feature_filename=args.meccano_rgb_feature_filename,
        ),
        "Val": GraphDatasetMeccano(
            metadata_root=str(root),
            split_name="Val",
            samples=val_samples,
            activity_to_idx=activity_to_idx,
            graph_type=graph_type,
            easg_cache_path=None,
            rgb_feature_filename=args.meccano_rgb_feature_filename,
        ),
        "Test": GraphDatasetMeccano(
            metadata_root=str(root),
            split_name="Test",
            samples=test_samples,
            activity_to_idx=activity_to_idx,
            graph_type=graph_type,
            easg_cache_path=None,
            rgb_feature_filename=args.meccano_rgb_feature_filename,
        ),
    }


def build_egtea_datasets(args, split_id, graph_type):
    input_root = Path(args.egtea_input_root)
    feature_root = Path(args.egtea_feature_root)
    annotations_root = Path(args.egtea_annotations_root)
    train_split_root = input_root / "train" / f"train_split_{split_id}"
    test_split_root = input_root / "test" / f"test_split_{split_id}"
    train_actions_path = annotations_root / f"train_split{split_id}.txt"
    test_actions_path = annotations_root / f"test_split{split_id}.txt"
    action_idx_path = annotations_root / "action_idx.txt"

    train_all_samples, _train_stats = collect_split_samples(
        str(train_split_root),
        str(feature_root),
        args.egtea_num_graphs,
        args.egtea_rgb_feature_filename,
        split_actions_path=str(train_actions_path),
        action_idx_path=str(action_idx_path),
    )
    test_samples, _test_stats = collect_split_samples(
        str(test_split_root),
        str(feature_root),
        args.egtea_num_graphs,
        args.egtea_rgb_feature_filename,
        split_actions_path=str(test_actions_path),
        action_idx_path=str(action_idx_path),
    )
    train_samples, val_samples = split_train_val(
        train_all_samples,
        args.egtea_val_fraction,
        val_num_samples=args.egtea_val_num_samples,
        reference_samples=test_samples
        if args.egtea_val_distribution_source == "test"
        else None,
    )
    all_samples = train_samples + val_samples + test_samples
    activity_to_idx = build_activity_mapping(all_samples)
    vocab = build_vocab(all_samples, str(input_root))
    dataset_kwargs = {
        "activity_to_idx": activity_to_idx,
        "graph_type": graph_type,
        "vocab": vocab,
        "clip_text_path": None,
        "easg_cache_path": None,
        "rgb_feature_filename": args.egtea_rgb_feature_filename,
    }
    return {
        "train": GraphDatasetEgtea(train_samples, **dataset_kwargs),
        "val": GraphDatasetEgtea(val_samples, **dataset_kwargs),
        "test": GraphDatasetEgtea(test_samples, **dataset_kwargs),
    }


def write_outputs(rows, output_prefix):
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)
    return csv_path, json_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure FASG/GPASG graph efficiency on MECCANO and EGTEA Gaze."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["meccano", "egtea"],
        default=["meccano", "egtea"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["fasg", "gpasg"],
        default=["fasg", "gpasg"],
    )
    parser.add_argument("--output-prefix", default="graph_efficiency_results")
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--meccano-root", default=str(DEFAULT_MECCANO_ROOT))
    parser.add_argument("--meccano-train-csv", default=None)
    parser.add_argument("--meccano-val-csv", default=None)
    parser.add_argument("--meccano-test-csv", default=None)
    parser.add_argument("--meccano-num-graphs", type=int, default=10)
    parser.add_argument(
        "--meccano-rgb-feature-filename",
        default="frame_features_model_dinov3_vits16.h5",
    )

    parser.add_argument("--egtea-input-root", default=str(DEFAULT_EGTEA_INPUT))
    parser.add_argument("--egtea-feature-root", default=str(DEFAULT_EGTEA_FEATURES))
    parser.add_argument(
        "--egtea-annotations-root", default=str(DEFAULT_EGTEA_ANNOTATIONS)
    )
    parser.add_argument("--egtea-split-ids", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--egtea-num-graphs", type=int, default=32)
    parser.add_argument("--egtea-val-fraction", type=float, default=0.1)
    parser.add_argument("--egtea-val-num-samples", type=int, default=None)
    parser.add_argument(
        "--egtea-val-distribution-source", choices=["test", "train"], default="test"
    )
    parser.add_argument(
        "--egtea-rgb-feature-filename",
        default="frame_features_model_dinov3_vitl16.h5",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []

    for model_name in args.models:
        graph_type = MODEL_TO_GRAPH_TYPE[model_name]

        if "meccano" in args.datasets:
            datasets = build_meccano_datasets(args, graph_type)
            metrics, sample_count = measure_datasets(
                datasets,
                f"MECCANO combined {model_name}",
                max_samples=args.max_samples,
            )
            rows.append(
                metrics.as_row(
                    "meccano",
                    "train_val_test",
                    None,
                    model_name,
                    graph_type,
                    sample_count,
                )
            )

        if "egtea" in args.datasets:
            for split_id in args.egtea_split_ids:
                datasets = build_egtea_datasets(args, split_id, graph_type)
                for split_name, dataset in datasets.items():
                    metrics, sample_count = measure_dataset(
                        dataset,
                        f"EGTEA split {split_id} {split_name} {model_name}",
                        max_samples=args.max_samples,
                    )
                    rows.append(
                        metrics.as_row(
                            "egtea_gaze",
                            split_name,
                            split_id,
                            model_name,
                            graph_type,
                            sample_count,
                        )
                    )

    csv_path, json_path = write_outputs(rows, args.output_prefix)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    for row in rows:
        print(
            "{dataset} split={split_id}:{split} model={model} "
            "graphs={graphs} avg_max_sp={avg_max_shortest_path:.4f} "
            "avg_eff={avg_global_efficiency:.4f} max_sp={max_shortest_path}".format(
                **row
            )
        )


if __name__ == "__main__":
    main()

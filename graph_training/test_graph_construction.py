import argparse
import random
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.GraphDataset import GraphDatasetMeccano
from dataset.meccano_aux import return_meccano_train_test_samples

DEFAULT_DATASET_ROOT = "/deepstore/datasets/dmb/ComputerVision/information_retrieval/MECCANO/updated_MECCANO_vlm_ann_Qwen3-VL-32B-Instruct-3fps"
DEFAULT_TRAIN_ACTIONS_CSV = "/deepstore/datasets/dmb/ComputerVision/information_retrieval/MECCANO/dataset/MECCANO_train_actions.csv"
DEFAULT_TEST_ACTIONS_CSV = "/deepstore/datasets/dmb/ComputerVision/information_retrieval/MECCANO/dataset/MECCANO_test_actions.csv"


def check_split(dataset, samples, split_name, num_examples, rng):
    num_examples = min(num_examples, len(samples))
    indices = sorted(rng.sample(range(len(samples)), num_examples))

    print(f"\nChecking {split_name} split on {num_examples} random examples")
    failures = []

    for idx in indices:
        sample = samples[idx]
        print(
            f"[{split_name}] idx={idx} clip={sample['clip_name']} "
            f"label={sample['label']} action_id={sample['action_id']}"
        )
        try:
            item = dataset[idx]
            for graph_idx, graph in item["full_action_graphs"].items():
                _ = graph.to_easg_tensors()
            print(
                f"  OK: built {len(item['full_action_graphs'])} graphs "
                f"for activity_label={int(item['activity_label'])}"
            )
        except Exception as exc:
            print(f"  FAIL: {type(exc).__name__}: {exc}")
            print(traceback.format_exc())
            failures.append(
                {
                    "index": idx,
                    "clip_name": sample["clip_name"],
                    "label": sample["label"],
                    "action_id": sample["action_id"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--train-actions-csv", default=DEFAULT_TRAIN_ACTIONS_CSV)
    parser.add_argument("--test-actions-csv", default=DEFAULT_TEST_ACTIONS_CSV)
    parser.add_argument("--graph-type", default="full", choices=["full", "pruned"])
    parser.add_argument("--num-graphs", type=int, default=10)
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_samples, test_samples, activity_to_idx, split_stats = return_meccano_train_test_samples(
        dataset_root=args.dataset_root,
        train_actions_csv=args.train_actions_csv,
        test_actions_csv=args.test_actions_csv,
        num_graphs=args.num_graphs,
    )

    print(f"Loaded MECCANO mapping with {len(activity_to_idx)} classes")
    print(f"Split stats: {split_stats}")

    train_dataset = GraphDatasetMeccano(
        f"{args.dataset_root}/Train",
        train_samples,
        activity_to_idx,
        args.graph_type,
    )
    test_dataset = GraphDatasetMeccano(
        f"{args.dataset_root}/Test",
        test_samples,
        activity_to_idx,
        args.graph_type,
    )

    rng = random.Random(args.seed)
    train_failures = check_split(
        train_dataset,
        train_samples,
        "train",
        args.num_examples,
        rng,
    )
    test_failures = check_split(
        test_dataset,
        test_samples,
        "test",
        args.num_examples,
        rng,
    )

    total_failures = train_failures + test_failures
    print("\nSummary")
    print(f"Train failures: {len(train_failures)}")
    print(f"Test failures: {len(test_failures)}")

    if total_failures:
        print("Failure details:")
        for failure in total_failures:
            print(f"  {failure}")
        raise SystemExit(1)

    print("All sampled graph constructions succeeded.")


if __name__ == "__main__":
    main()

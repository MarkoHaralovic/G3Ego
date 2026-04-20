import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.GraphDataset import GraphDatasetMeccano
from dataset.meccano_aux import return_meccano_train_test_samples

DATASET_ROOT = "/deepstore/datasets/dmb/ComputerVision/information_retrieval/MECCANO/updated_MECCANO_vlm_ann_Qwen3-VL-32B-Instruct-3fps"
TRAIN_ACTIONS_CSV = "/deepstore/datasets/dmb/ComputerVision/information_retrieval/MECCANO/dataset/MECCANO_train_actions.csv"
TEST_ACTIONS_CSV = "/deepstore/datasets/dmb/ComputerVision/information_retrieval/MECCANO/dataset/MECCANO_test_actions.csv"
NUM_GRAPHS = 10
GRAPH_TYPE = "full"


def main():
    train_samples, test_samples, activity_to_idx, split_stats = return_meccano_train_test_samples(
        dataset_root=DATASET_ROOT,
        train_actions_csv=TRAIN_ACTIONS_CSV,
        test_actions_csv=TEST_ACTIONS_CSV,
        num_graphs=NUM_GRAPHS,
    )

    train_dataset = GraphDatasetMeccano(
        DATASET_ROOT,
        train_samples,
        activity_to_idx,
        GRAPH_TYPE,
    )
    test_dataset = GraphDatasetMeccano(
        DATASET_ROOT,
        test_samples,
        activity_to_idx,
        GRAPH_TYPE,
    )

    print(f"Official MECCANO classes: {len(activity_to_idx)}")
    print(f"First 10 official mappings: {list(activity_to_idx.items())[:10]}")
    print(f"Split stats: {split_stats}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    train_item = train_dataset[0]
    test_item = test_dataset[0]

    print(
        "Train sample:",
        train_item["clip_name"],
        train_item["activity_name"],
        int(train_item["activity_label"]),
        len(train_item["full_action_graphs"]),
    )
    print(
        "Test sample:",
        test_item["clip_name"],
        test_item["activity_name"],
        int(test_item["activity_label"]),
        len(test_item["full_action_graphs"]),
    )


if __name__ == "__main__":
    main()

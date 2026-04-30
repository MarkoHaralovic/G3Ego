import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from meccano_ablation_utils import load_json, run_ablation, selected_run_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument(
        "--graph-count", type=int, choices=[1, 3, 5, 16, 32], required=True
    )
    args = parser.parse_args()

    ablation_config = load_json(args.config_path)
    config = selected_run_config(ablation_config, graph_count=args.graph_count)

    print(
        "Running MECCANO action-recognition graph-count ablation: "
        f"{ablation_config['model_type']}, {config['data']['graph_type']}, "
        f"{args.graph_count} graph(s)"
    )
    run_ablation(config, ablation_config)


if __name__ == "__main__":
    main()

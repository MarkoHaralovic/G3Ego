import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from meccano_ablation_utils import (
    LOSS_CONFIGS,
    load_json,
    run_ablation,
    selected_run_config,
)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--loss-name", choices=sorted(LOSS_CONFIGS), required=True)
    parser.add_argument(
        "--graph-count", type=int, choices=[5, 10, 16, 32], default=None
    )
    args = parser.parse_args()

    ablation_config = load_json(args.config_path)
    config = selected_run_config(
        ablation_config,
        graph_count=args.graph_count,
        loss_name=args.loss_name,
    )

    print(
        "Running MECCANO action-recognition loss ablation: "
        f"{ablation_config['model_type']}, {config['data']['graph_type']}, "
        f"{config['mlp']['num_graphs']} graph(s), {args.loss_name}"
    )
    run_ablation(config, ablation_config)


if __name__ == "__main__":
    main()

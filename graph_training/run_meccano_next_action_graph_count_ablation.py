import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from meccano_ablation_utils import load_json, run_ablation, source_run_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--graph-count", type=int, choices=[1, 3, 5], required=True)
    parser.add_argument("--num-epochs", type=int, default=20)
    args = parser.parse_args()

    ablation_config = load_json(args.config_path)
    config = source_run_config(
        ablation_config,
        graph_count=args.graph_count,
        num_epochs=args.num_epochs,
    )

    print(
        "Running MECCANO next-action graph-count ablation: "
        f"{ablation_config['model_type']}, {config['data']['graph_type']}, "
        f"{args.graph_count} graph(s), {args.num_epochs} epoch(s)"
    )
    run_ablation(config, ablation_config)


if __name__ == "__main__":
    main()

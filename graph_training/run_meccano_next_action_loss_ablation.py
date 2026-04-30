import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from meccano_ablation_utils import (
    NEXT_ACTION_LOSSES,
    load_json,
    run_ablation,
    source_run_config,
)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--loss-name", choices=NEXT_ACTION_LOSSES, required=True)
    parser.add_argument("--graph-count", type=int, default=10)
    parser.add_argument("--num-epochs", type=int, default=20)
    args = parser.parse_args()

    ablation_config = load_json(args.config_path)
    config = source_run_config(
        ablation_config,
        graph_count=args.graph_count,
        loss_name=args.loss_name,
        num_epochs=args.num_epochs,
    )

    print(
        "Running MECCANO next-action loss ablation: "
        f"{ablation_config['model_type']}, {config['data']['graph_type']}, "
        f"{args.graph_count} graph(s), {args.num_epochs} epoch(s), {args.loss_name}"
    )
    run_ablation(config, ablation_config)


if __name__ == "__main__":
    main()

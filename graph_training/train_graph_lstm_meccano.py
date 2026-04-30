import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = json.load(f)

    from train_graph_lstm_meccano_optuna import train_one_run

    train_one_run(config, trial=None, run_test=True)


if __name__ == "__main__":
    main()

# MECCANO Action Recognition Experiments

This folder mirrors the next-action experiment layout for action recognition.

## Optuna HPO

Base configs:

- `optuna_base/mlp_full.json`
- `optuna_base/mlp_pruned.json`
- `optuna_base/lstm_full.json`
- `optuna_base/lstm_pruned.json`

Search-space configs:

- `../optuna/action_recognition/mlp_full.json`
- `../optuna/action_recognition/mlp_pruned.json`
- `../optuna/action_recognition/lstm_full.json`
- `../optuna/action_recognition/lstm_pruned.json`

Slurm scripts:

- `slurm_scripts/meccano/action_recognition/optuna/mlp_full.sbatch`
- `slurm_scripts/meccano/action_recognition/optuna/mlp_pruned.sbatch`
- `slurm_scripts/meccano/action_recognition/optuna/lstm_full.sbatch`
- `slurm_scripts/meccano/action_recognition/optuna/lstm_pruned.sbatch`

## Balanced Training

Balanced focal-loss configs are in `balanced_training/`. These are ready to run directly
and save under `outputs/meccano/action_recognition/balanced_training/`.

## Trainable Fusion

Trainable fusion configs are in `trainable_fusion/`. Run the matching balanced training
job first, because these configs discover the latest `best_model_top5_epoch_*.pt`
checkpoint from the balanced output directory.

## Graph Count Ablation

Configs in `graph_count_ablation/` reuse the selected top-1 Optuna hyperparameters
from `outputs/meccano/action_recognition/selected_top1_20/`, override only
`mlp.num_graphs`, and train/test fresh models for 1, 3, and 5 graph frames.

Run all LSTM/MLP full/pruned combinations with:

```bash
sbatch slurm_scripts/meccano/action_recognition/graph_count_ablation/all.sbatch
```

Outputs are written under
`outputs/meccano/action_recognition/graph_count_ablation/`.

## Loss Ablation

Configs in `loss_ablation/` reuse the selected top-1 Optuna hyperparameters
from `outputs/meccano/action_recognition/selected_top1_20/`, keep the selected
10-graph setup, and override only the training loss.

Run CE and annealed focal loss for all LSTM/MLP full/pruned combinations with:

```bash
sbatch slurm_scripts/meccano/action_recognition/loss_ablation/all.sbatch
```

The annealed focal setting uses `gamma=2.0` at the first epoch and exponentially
decays it to `0.1` by the last epoch. Outputs are written under
`outputs/meccano/action_recognition/loss_ablation/`.

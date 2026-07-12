#!/bin/bash
#SBATCH --partition=main,students
#SBATCH --job-name=extract_meccano_lstm_ohg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/extract_meccano_lstm_ohg_%J.log

set -euo pipefail

RUN_ROOT="${1:-/home/s3758869/egocentric_video_graph_framework_ar/outputs/meccano/graph_lstm_next_action_ohg_ablation/graph_lstm_next_action_pruned_delta1s_rgb_ohg/dino_model_fc_layer_2_num_epoch_20_graph_emb_dim_512_lstm_hidden_64_final_graph_emb_dim_64_20260612_184411}"
CHECKPOINT_METRIC="${2:-top5}"
OUTPUT_ROOT="${3:-/home/s3758869/tempAgg/data/meccano}"
BATCH_SIZE="${4:-16}"
NUM_WORKERS="${5:-2}"
FEATURE_KIND="${6:-graph_lstm_hidden}"

mkdir -p /home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs

source /home/s3758869/miniconda3/etc/profile.d/conda.sh
conda activate ego_graphs
export PYTHONNOUSERSITE=1

cd /home/s3758869/tempAgg

python extract_meccano_lstm_tempagg_features.py \
  --run-root "${RUN_ROOT}" \
  --checkpoint-metric "${CHECKPOINT_METRIC}" \
  --feature-kind "${FEATURE_KIND}" \
  --output-root "${OUTPUT_ROOT}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --overwrite

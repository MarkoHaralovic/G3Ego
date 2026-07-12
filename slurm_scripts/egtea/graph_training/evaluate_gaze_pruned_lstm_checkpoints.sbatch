#!/bin/bash
#SBATCH --job-name=egtea_eval_lstm_ckpts
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --partition=main,students
#SBATCH --output=/home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/egtea_eval_lstm_ckpts_%j.out

set -euo pipefail

source /home/s3758869/miniconda3/etc/profile.d/conda.sh
conda activate ego_graphs

cd /home/s3758869/egocentric_video_graph_framework_ar/graph_training

BASELINE_DIR="/home/s3758869/egocentric_video_graph_framework_ar/outputs/egtea_gaze/action_recognition/lstm_per_split_gaze_pruned_testbest_50ep/egtea_gaze_pruned_lstm_split1_30ep/egtea_lstm_cross_split_20260612_211708/split_1/dino_model_fc_layer_2_num_epoch_30_graph_emb_dim_512_lstm_hidden_128_final_graph_emb_dim_256"
OHD_DIR="/home/s3758869/egocentric_video_graph_framework_ar/outputs/egtea_gaze/action_recognition/lstm_per_split_gaze_pruned_ohd_30ep/egtea_gaze_pruned_ohd_lstm_split1_30ep/egtea_lstm_cross_split_20260614_173544/split_1/dino_model_fc_layer_2_num_epoch_30_graph_emb_dim_512_lstm_hidden_128_final_graph_emb_dim_256"
OUT_DIR="/home/s3758869/egocentric_video_graph_framework_ar/outputs/egtea_gaze/action_recognition/checkpoint_test_eval"

python evaluate_lstm_checkpoints_egtea.py \
  --checkpoint-dir "$BASELINE_DIR" \
  --checkpoint-dir "$OHD_DIR" \
  --output-dir "$OUT_DIR" \
  --num-workers 4

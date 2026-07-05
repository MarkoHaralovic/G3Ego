#!/bin/bash
#SBATCH --partition=main,students
#SBATCH --job-name=slowfast_eval
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --output /home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/slowfast_eval%J.log

cd /home/s3758869
source  source /aria_env/bin/activate

cd /home/s3758869/egocentric_video_graph_framework_ar/SlowFast
python tools/run_net.py --cfg configs/MECCANO/configs/action_recognition/SLOWFAST_8x8_R50_MECCANO.yaml
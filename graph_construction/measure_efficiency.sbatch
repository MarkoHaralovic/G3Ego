#!/bin/bash
#SBATCH --job-name=graph_efficiency
#SBATCH --partition=main,students
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/s3758869/slurm_outputs/graph_efficiency_%j.out

set -euo pipefail

mkdir -p /home/s3758869/slurm_outputs

source /home/s3758869/miniconda3/etc/profile.d/conda.sh
conda activate ego_graphs

cd /projects/eemcs/dmb/ComputerVision/ego_graphs/marko/egocentric_video_graph_framework_ar

PYTHONPYCACHEPREFIX=/tmp/pycache_graph_eff_${SLURM_JOB_ID} \
python graph_construction/measure_efficiency.py \
  --datasets meccano egtea \
  --models fasg gpasg \
  --output-prefix /home/s3758869/slurm_outputs/graph_efficiency_${SLURM_JOB_ID}

#!/bin/bash
#SBATCH --partition=short
#SBATCH --job-name=parse_test
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --output=/home/s3758869/egocentric_video_graph_framework_ar/test_parse_output.txt

export PYTHONNOUSERSITE=1
/home/s3758869/miniconda3/envs/ego_graphs/bin/python /home/s3758869/egocentric_video_graph_framework_ar/test_parse_fix.py

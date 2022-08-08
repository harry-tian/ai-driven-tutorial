#!/bin/bash
#
#SBATCH --mail-user=tianh@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/tianh/slurm/out/%j.%N.stdout
#SBATCH --error=/home/tianh/slurm/stderr/%j.%N.stderr
#SBATCH --job-name=triplets
#SBATCH --partition=cdac-contrib
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --exclude=aa[001-002]

DATA=wv_linear_mm

python MTL.py \
            --dataset_config=configs/wv_3d_square/dataset.yaml \
            --model_config=configs/models/MTL0.5.yaml \
            --triplet_config=configs/wv_3d_square/triplets/filtered/aligns/align=0.5_filtered.yaml \
            --overwrite_config=configs/filtered.yaml \
            --embed_dim=50

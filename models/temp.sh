#!/bin/bash
#
#SBATCH --mail-user=tianh@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/tianh/slurm/out/%j.%N.stdout
#SBATCH --error=/home/tianh/slurm/stderr/%j.%N.stderr
#SBATCH --job-name=triplets
#SBATCH --partition=cdac-own
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=10G

python TN.py \
        --dataset_config=configs/bm.yaml \
        --model_config=configs/models/TN.yaml \
        --triplet_config=configs/bm/triplets/lpips.yaml \
        --overwrite_config=configs/overwrite.yaml \
        --embed_dim=512 \

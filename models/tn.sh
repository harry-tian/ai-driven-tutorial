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

for i in {0..0}
        do python TN.py \
                --dataset_config=configs/bird.yaml \
                --model_config=configs/models/TN.yaml \
                --triplet_config=configs/bird/triplets/lpips.yaml \
                --overwrite_config=configs/filtered.yaml \
                --embed_dim=50 \
                --seed=$i 
done
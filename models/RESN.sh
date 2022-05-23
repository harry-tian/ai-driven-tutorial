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

for seed in {0..2}
        do python RESN.py \
        --dataset_config=configs/wv_3d_linear/dataset.yaml \
        --model_config=configs/wv_3d_linear/models/RESN.yaml \
        --triplet_config=configs/wv_3d_linear/triplets/filtered/align=0.5_filtered.yaml \
        --overwrite_config=configs/wv_3d_linear/overwrite.yaml \
        --seed=$seed \
        --out_csv=wv_3d_linear_RESN.csv \
        --wandb_group=wv_3d_linear_RESN 
done


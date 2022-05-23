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

for seed in {0..2}
        do python RESN.py \
        --dataset_config=configs/wv_3d_blob/dataset.yaml \
        --model_config=configs/models/RESN.yaml \
        --triplet_config=configs/wv_3d/triplets/filtered/aligns/align=0.5_filtered.yaml \
        --overwrite_config=configs/wv_3d_blob/overwrite.yaml \
        --seed=$seed \
        --wandb_project=wv_3d_blob_RESN \
        --embeds_output_dir=../embeds/wv_3d_blob_RESN
done


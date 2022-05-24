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

DATA=wv_3d_linear0

python RESN.py \
                --dataset_config=configs/$DATA/dataset.yaml \
                --model_config=configs/models/RESN.yaml \
                --triplet_config=configs/$DATA/triplets/filtered/aligns/align=1_filtered.yaml \
                --overwrite_config=configs/$DATA/overwrite.yaml \
                --seed=$2 \
                --wandb_project=$DATA"_RESN" \
                --embeds_output_dir=$DATA"_RESN" \
                --embed_dim=$1

# DIMS=(50 512); for i in {0..1}; do for seed in {0..2}; do sbatch RESN.sh "${DIMS[i]}" $seed; done; done;

# DIMS=(50 512)
# for i in {0..1}
#         do for seed in {0..2}
#                 do python RESN.py \
#                 --dataset_config=configs/$DATA/dataset.yaml \
#                 --model_config=configs/models/RESN.yaml \
#                 --triplet_config=configs/$DATA/triplets/filtered/aligns/align=0.5_filtered.yaml \
#                 --overwrite_config=configs/$DATA/overwrite.yaml \
#                 --seed=$seed \
#                 --wandb_project=$DATA"_RESN" \
#                 --embeds_output_dir=$DATA"_RESN" \
#                 --embed_dim="${DIMS[i]}"
#         done
# done


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
#SBATCH --exclude=aa[001-002]

 
python MTL.py \
    --dataset_config=configs/wv_3d/dataset.yaml \
    --model_config=configs/wv_3d/models/MTL0.5.yaml \
    --triplet_config=configs/wv_3d/align_triplets/align=0.8.yaml \
    --overwrite_config=configs/wv_3d/overwrite.yaml \
    --seed=0

    
 
# python MTL.py \
#     --dataset_config=configs/wv_3d/dataset.yaml \
#     --model_config=configs/wv_3d/models/MTL0.5.yaml \
#     --triplet_config=configs/wv_3d/align_triplets/align=0.8.yaml \
#     --overwrite_config=configs/wv_3d/overwrite.yaml \
#     --seed=1
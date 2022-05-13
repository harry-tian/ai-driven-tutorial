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

# for i in {0..4}
#       do python MTL.py \
#       --dataset_config=configs/wv_3d/dataset.yaml \
#       --model_config=configs/wv_3d/models/RESN.yaml \
#       --triplet_config=configs/wv_3d/align_triplets/align=0.5.yaml \
#       --overwrite_config=configs/wv_3d/overwrite.yaml \
#       --seed=$i
# done

    
python MTL_han.py \
    --dataset_config=configs/wv_3d/dataset.yaml \
    --model_config=configs/wv_3d/models/TN.yaml \
    --triplet_config=configs/wv_3d/align_triplets/align=0.8.yaml \
    --overwrite_config=configs/wv_3d/overwrite.yaml \
    --seed=0

# python MTL_han.py \
#     --dataset_config=configs/wv_3d/dataset.yaml \
#     --model_config=configs/wv_3d/models/MTL0.5.yaml \
#     --triplet_config=configs/wv_3d/filtered_triplets/align=0.8_filtered.yaml \
#     --overwrite_config=configs/wv_3d/overwrite.yaml \
#     --seed=0

# python MTL_han.py \
#     --dataset_config=configs/wv_3d/dataset.yaml \
#     --model_config=configs/wv_3d/models/MTL0.5.yaml \
#     --triplet_config=configs/wv_3d/noisy_triplets/noise=0.1.yaml \
#     --overwrite_config=configs/wv_3d/overwrite.yaml \
#     --seed=0
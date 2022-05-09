#!/bin/bash
#
#SBATCH --mail-user=tianh@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/tianh/slurm/out/%j.%N.stdout
#SBATCH --error=/home/tianh/slurm/stderr/%j.%N.stderr
#SBATCH --job-name=TN_syn
#SBATCH --partition=cdac-contrib
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=50Gs
#SBATCH --exclude=aa[001-002]

python MTL_han.py \
    --dataset_config=configs/bm/dataset.yaml \
    --model_config=configs/bm/MTL.yaml \
    --triplet_config=configs/bm/triplets.yaml \

# for i in {0..5}
#     do python MTL_han.py \
#         --dataset_config=configs/bm/dataset.yaml \
#         --model_config=configs/bm/MTL.yaml \
#         --triplet_config=configs/bm/triplets.yaml \
#         --seed=$i \
#         --lamda=0 \
#         --wandb_name=MTL0s$i
# done

# for i in {0..5}
#     do python MTL_han.py \
#         --dataset_config=configs/bm/dataset.yaml \
#         --model_config=configs/bm/MTL.yaml \
#         --triplet_config=configs/bm/triplets.yaml \
#         --seed=$i \
#         --lamda=1 \
#         --wandb_name=MTL1s$i
# done

# for i in {0..5}
#     do python MTL_han.py \
#         --dataset_config=configs/bm/dataset.yaml \
#         --model_config=configs/bm/MTL.yaml \
#         --triplet_config=configs/bm/triplets.yaml \
#         --seed=$i \
#         --lamda=0.5 \
#         --wandb_name=MTL0.5s$i
# done

# for i in {0..5}
#     do python MTL_han.py \
#         --dataset_config=configs/bm/dataset.yaml \
#         --model_config=configs/bm/MTL.yaml \
#         --triplet_config=configs/bm/triplets.yaml \
#         --seed=$i \
#         --lamda=0.2 \
#         --wandb_name=MTL0.2s$i
# done

# for i in {0..5}
#     do python MTL_han.py \
#         --dataset_config=configs/bm/dataset.yaml \
#         --model_config=configs/bm/MTL.yaml \
#         --triplet_config=configs/bm/triplets.yaml \
#         --seed=$i \
#         --lamda=0.8 \
#         --wandb_name=MTL0.8s$i
# done
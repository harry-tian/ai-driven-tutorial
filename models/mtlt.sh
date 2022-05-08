#!/bin/bash
#
#SBATCH --mail-user=tianh@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/tianh/slurm/out/%j.%N.stdout
#SBATCH --error=/home/tianh/slurm/stderr/%j.%N.stderr
#SBATCH --job-name=TN_syn
#SBATCH --partition=cdac-own
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=8Gs


# python MTLT.py \
#     --dataset_config=configs/bm/dataset.yaml \
#     --model_config=configs/bm/MTLT.yaml \
#     --triplet_config=configs/bm/triplets.yaml \

for i in {0..9}
    do python MTLT.py \
        --dataset_config=configs/wv_2d/dataset.yaml \
        --model_config=configs/wv_2d/models/MTLT.yaml \
        --triplet_config=configs/wv_2d/triplets/align=0.7.yaml \
        --seed=$i \
        --wandb_name=MTLT0.5s$i
done

for i in {0..9}
    do python MTLT.py \
        --dataset_config=configs/wv_2d/dataset.yaml \
        --model_config=configs/wv_2d/models/MTLT.yaml \
        --triplet_config=configs/wv_2d/triplets/align=0.7.yaml \
        --seed=$i \
        --lamda=0.2 \
        --wandb_name=MTLT0.2s$i
done

for i in {0..9}
    do python MTLT.py \
        --dataset_config=configs/wv_2d/dataset.yaml \
        --model_config=configs/wv_2d/models/MTLT.yaml \
        --triplet_config=configs/wv_2d/triplets/align=0.7.yaml \
        --seed=$i \
        --lamda=0.8 \
        --wandb_name=MTLT0.8s$i
done
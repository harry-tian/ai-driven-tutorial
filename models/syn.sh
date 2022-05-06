#!/bin/bash
#
#SBATCH --mail-user=tianh@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/tianh/slurm/out/%j.%N.stdout
#SBATCH --error=/home/tianh/slurm/stderr/%j.%N.stderr
#SBATCH --job-name=TN_syn
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=10Gs

python TN.py \
    --model_config=configs/bm/model.yaml \
    --dataset_config=configs/bm/dataset.yaml \
    --triplet_config=configs/bm/triplets.yaml \


# python RESN.py \
#     --model_config=configs/wv_2d/model.yaml \
#     --dataset_config=configs/wv_2d/dataset.yaml \
#     --triplet_config=configs/wv_2d/w1=2.7303.yaml \

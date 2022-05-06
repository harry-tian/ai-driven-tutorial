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
#SBATCH --mem=8G


# python RESN.py --model_config=configs/wv_3d/true_label.yaml 

# python RESN.py \
#     --model_config=configs/wv_2d/RESN.yaml \
#     --dataset_config=configs/wv_2d/dataset.yaml \
#     --triplet_config=configs/wv_2d/w1=1_w2=0.yaml \


python MTL.py \
    --model_config=configs/wv_2d/MTL.yaml \
    --dataset_config=configs/wv_2d/dataset.yaml \
    --triplet_config=configs/wv_2d/w1=1_w2=0.yaml \

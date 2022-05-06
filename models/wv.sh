#!/bin/bash
#
#SBATCH --mail-user=tianh@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/tianh/slurm/out/%j.%N.stdout
#SBATCH --error=/home/tianh/slurm/stderr/%j.%N.stderr
#SBATCH --job-name=triplets
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --exclude=aa[001-002]
  
# export MODEL=RESN

# python main.py \
#     --dataset_config=configs/wv_2d/dataset.yaml \
#     --model_config=$@ \
#     --triplet_config=configs/wv_2d/triplets/w1=1_w2=0.yaml \

for i in {0..9}
    do python main.py \
            --dataset_config=configs/wv_2d/dataset.yaml \
            --model_config=configs/wv_2d/models/RESN.yaml \
            --triplet_config=configs/wv_2d/triplets/w1=1_w2=0.yaml \
            --seed=$i
done
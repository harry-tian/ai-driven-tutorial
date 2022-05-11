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

python gen_embs.py \
    --dataset=wv \
    --model_name=MTL_han \
    --subdir= \
    --wandb_group=wv_2d \
    --wandb_name=RESN \
    --wandb_run=3tms2l1f \
    --suffix=seed0

python gen_embs.py \
    --dataset=wv \
    --model_name=MTL_han \
    --subdir= \
    --wandb_group=wv_2d \
    --wandb_name=RESN \
    --wandb_run=g43165ei \
    --suffix=seed1

python gen_embs.py \
    --dataset=wv \
    --model_name=MTL_han \
    --subdir= \
    --wandb_group=wv_2d \
    --wandb_name=RESN \
    --wandb_run=35sfyggl \
    --suffix=seed2

python gen_embs.py \
    --dataset=wv \
    --model_name=MTL_han \
    --subdir= \
    --wandb_group=wv_2d \
    --wandb_name=RESN \
    --wandb_run=2xqbs692 \
    --suffix=seed3
    
python gen_embs.py \
    --dataset=wv \
    --model_name=MTL_han \
    --subdir= \
    --wandb_group=wv_2d \
    --wandb_name=RESN \
    --wandb_run=3f17ht03 \
    --suffix=seed4

python gen_embs.py \
    --dataset=wv \
    --model_name=MTL_han \
    --subdir= \
    --wandb_group=wv_2d \
    --wandb_name=RESN \
    --wandb_run=3ajj2lcv \
    --suffix=seed5
    
python gen_embs.py \
    --dataset=wv \
    --model_name=MTL_han \
    --subdir= \
    --wandb_group=wv_2d \
    --wandb_name=RESN \
    --wandb_run=30kjjz9b \
    --suffix=seed6
    
python gen_embs.py \
    --dataset=wv \
    --model_name=MTL_han \
    --subdir= \
    --wandb_group=wv_2d \
    --wandb_name=RESN \
    --wandb_run=36gi91ok \
    --suffix=seed8
    
python gen_embs.py \
    --dataset=wv \
    --model_name=MTL_han \
    --subdir= \
    --wandb_group=wv_2d \
    --wandb_name=RESN \
    --wandb_run=8120u09x \
    --suffix=seed9
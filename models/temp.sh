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
    --wandb_run=1tip7ey5 \
    --suffix=seed3


#!/bin/bash
#
#SBATCH --mail-user=tianh@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/tianh/slurm/out/%j.%N.stdout
#SBATCH --error=/home/tianh/slurm/stderr/%j.%N.stderr
#SBATCH --job-name=triplets
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=20000
#SBATCH --nodelist=aa[003]

hostname
echo $CUDA_VISIBLE_DEVICES

python TN_food.py \
  --split_by=img \
  --img_split=0.8 \
  --wandb_project=triplet_net_food_img \
  --wandb_group=split=0.8_lr=1e-4 \
  --max_epochs=5 \
  --learning_rate=1e-4 \
  --train_batch_size=100 \
  --do_train \
  --do_test


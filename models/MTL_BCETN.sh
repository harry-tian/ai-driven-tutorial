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
#SBATCH --nodelist=aa[001]

hostname
echo $CUDA_VISIBLE_DEVICES

python MTL_BCETN.py \
  --embed_dim=10 \
  --wandb_project=MTL_2 \
  --wandb_group=lambdas \
  --wandb_mode=online \
  --pretrained \
  --max_epochs=48 \
  --learning_rate=1e-4 \
  --train_batch_size=64 \
  --do_train \
  --do_test \
  --subset \
  --lamda=0.1


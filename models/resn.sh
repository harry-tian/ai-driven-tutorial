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
#SBATCH --nodelist=aa002
  
python resn_args.py \
  --wandb_mode=online \
  --wandb_entity=ai-driven-tutorial \
  --wandb_project=baselines \
  --wandb_group=bird \
  --wandb_name=bs=32 \
  --train_dir=/net/scratch/tianh-shared/bird/train \
  --valid_dir=/net/scratch/tianh-shared/bird/valid \
  --test_dir=/net/scratch/tianh-shared/bird/valid \
  --num_class=4 \
  --transform=bm \
  --embed_dim=10 \
  --max_epochs=200 \
  --train_batch_size=32 \
  --learning_rate=1e-4 \
  --pretrained \
  --do_train \
  --do_test
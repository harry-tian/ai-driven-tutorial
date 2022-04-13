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
  
python resn_px.py \
  --wandb_mode=online \
  --wandb_entity=ai-driven-tutorial \
  --wandb_project=tests \
  --wandb_group=prostatex \
  --wandb_name=bs=32 \
  --train_dir=/net/scratch/tianh-shared/prostatex/PZ_npys/auto_split/train \
  --valid_dir=/net/scratch/tianh-shared/prostatex/PZ_npys/auto_split/valid \
  --test_dir=/net/scratch/tianh-shared/prostatex/PZ_npys/auto_split/test \
  --num_class=2 \
  --transform=xray \
  --embed_dim=10 \
  --max_epochs=200 \
  --train_batch_size=32 \
  --learning_rate=1e-4 \
  --do_train \
  --do_test
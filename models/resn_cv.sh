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
  
python resn_cv.py \
  --wandb_mode=online \
  --wandb_entity=ai-driven-tutorial \
  --wandb_project=warblers \
  --wandb_group=cv \
  --wandb_name=split3 \
  --split_idx=3 \
  --train_dir=/net/scratch/tianh-shared/misc/inat/4class \
  --splits=bird_splits.pkl \
  --num_class=4 \
  --embed_dim=9 \
  --pretrained \
  --max_epochs=200 \
  --train_batch_size=128 \
  --learning_rate=1e-4 \
  --transform=xray \
  --do_train \
  --do_test \
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
#SBATCH --nodelist=aa001

python TN.py \
  --wandb_mode=online \
  --wandb_project=bm-htriplets \
  --wandb_group=unpretrained \
  --wandb_name=TN \
  --train_dir=/net/scratch/hanliu-shared/data/bm/train \
  --valid_dir=/net/scratch/hanliu-shared/data/bm/valid \
  --test_dir=/net/scratch/hanliu-shared/data/bm/valid \
  --train_triplets=/net/scratch/tianh/explain_teach/data/bm_triplets/3c2_unique=182/train_triplets.pkl \
  --valid_triplets=/net/scratch/tianh/explain_teach/data/bm_triplets/3c2_unique=182/valid_triplets.pkl \
  --num_class=2 \
  --embed_dim=10 \
  --max_epochs=200 \
  --learning_rate=1e-4 \
  --train_batch_size=160 \
  --transform=bm \
  --do_train 
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
#SBATCH --mem=10G
#SBATCH --nodelist=b003
  
CUDA_LAUNCH_BLOCKING=1 python RESN.py \
  --wandb_mode=online \
  --wandb_entity=ai-driven-tutorial \
  --wandb_project=baselines \
  --wandb_group=bm \
  --wandb_name=resm \
  --train_dir=/net/scratch/tianh-shared/bm/train \
  --valid_dir=/net/scratch/tianh-shared/bm/valid \
  --test_dir=/net/scratch/tianh-shared/bm/test \
  --train_triplets=/net/scratch/tianh/explain_teach/data/bm_triplets/3c2_unique=182/train_triplets_120.pkl \
  --valid_triplets=/net/scratch/tianh/explain_teach/data/bm_triplets/3c2_unique=182/test_triplets.pkl \
  --test_triplets=/net/scratch/tianh/explain_teach/data/bm_triplets/3c2_unique=182/fake_valid_triplets.pkl \
  --num_class=2 \
  --embed_dim=10 \
  --max_epochs=100 \
  --learning_rate=1e-4 \
  --train_batch_size=32 \
  --transform=bm \
  --do_train \
  --do_test
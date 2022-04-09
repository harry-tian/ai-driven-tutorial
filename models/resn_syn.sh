#!/bin/bash
#
#SBATCH --mail-user=chacha@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/chacha/slurm/out/%j.%N.stdout
#SBATCH --error=/home/chacha/slurm/stderr/%j.%N.stderr
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
  --wandb_project=resn \
  --wandb_group=bird \
  --wandb_name=bs=32 \
  --train_dir=/net/scratch/chacha/data/weevil_vespula/train \
  --valid_dir=/net/scratch/chacha/data/weevil_vespula/valid \
  --test_dir=/net/scratch/chacha/data/weevil_vespula/test \
  --num_class=2 \
  --transform=wv \
  --embed_dim=10 \
  --max_epochs=200 \
  --train_batch_size=32 \
  --learning_rate=1e-4 \
  --pretrained \
  --do_train \
  --do_test
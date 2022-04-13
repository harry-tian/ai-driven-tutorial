#!/bin/bash
#
#SBATCH --mail-user=chacha@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/chacha/slurm/out/%j.%N.stdout
#SBATCH --error=/home/chacha/slurm/stderr/%j.%N.stderr
#SBATCH --job-name=MTL_syn_lambda_1
#SBATCH --partition=cdac-own
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=20000
#SBATCH --nodelist=a006

python MTL_RESNTN.py \
  --wandb_mode=online \
  --wandb_project=synthetic_MTL \
  --wandb_group=unpretrained \
  --wandb_name=MTL_syn_lambda_1 \
  --train_dir=/net/scratch/chacha/data/weevil_vespula/train \
  --valid_dir=/net/scratch/chacha/data/weevil_vespula/valid \
  --test_dir=/net/scratch/chacha/data/weevil_vespula/test \
  --train_triplets=/net/scratch/chacha/data/weevil_vespula/train_triplet.pkl \
  --valid_triplets=/net/scratch/chacha/data/weevil_vespula/valid_triplet.pkl \
  --test_triplets=/net/scratch/chacha/data/weevil_vespula/test_triplet.pkl \
  --num_class=2 \
  --embed_dim=10 \
  --max_epochs=200 \
  --learning_rate=1e-4 \
  --train_batch_size=120 \
  --lamda=1 \
  --transform=wv \
  --do_train \
  --do_test
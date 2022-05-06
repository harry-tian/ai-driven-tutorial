#!/bin/bash
#
#SBATCH --mail-user=chacha@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/chacha/slurm/out/%j.%N.stdout
#SBATCH --error=/home/chacha/slurm/stderr/%j.%N.stderr
#SBATCH --job-name=MTL_syn_lambda_0.8
#SBATCH --partition=cdac-contrib
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=20000
#SBATCH --nodelist=a008

python MTL.py \
  --wandb_mode=online \
  --wandb_project=tests \
  --wandb_group=bm \
  --wandb_name=MTL \
  --train_dir=/net/scratch/chacha-shared/weevil_vespula/train \
  --valid_dir=/net/scratch/chacha-shared/weevil_vespula/valid \
  --test_dir=/net/scratch/chacha-shared/weevil_vespula/test \
  --train_triplets=/net/scratch/chacha-shared/weevil_vespula/train_triplet.pkl \
  --valid_triplets=/net/scratch/chacha-shared/weevil_vespula/valid_triplet.pkl \
  --test_triplets=/net/scratch/chacha-shared/weevil_vespula/test_triplet.pkl \
  --num_class=2 \
  --embed_dim=10 \
  --max_epochs=100 \
  --learning_rate=1e-4 \
  --train_batch_size=120 \
  --syn \
  --train_synthetic=../embeds/wv/train.pkl \
  --test_synthetic=../embeds/wv/test.pkl \
  --w1=2.73027025 \
  --w2=1 \
  --lamda=0.5 \
  --transform=wv \
  --dataloader_num_workers=2 \
  --do_train \
  --do_test
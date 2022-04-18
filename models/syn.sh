#!/bin/bash
#
#SBATCH --mail-user=tianh@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/tianh/slurm/out/%j.%N.stdout
#SBATCH --error=/home/tianh/slurm/stderr/%j.%N.stderr
#SBATCH --job-name=TN_syn
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --nodelist=c001


python MTL.py \
  --wandb_mode=online \
  --wandb_project=wv \
  --wandb_group=w1=10000 \
  --wandb_name=MTL_l=0.8 \
  --train_dir=/net/scratch/chacha-shared/weevil_vespula/train \
  --valid_dir=/net/scratch/chacha-shared/weevil_vespula/valid \
  --test_dir=/net/scratch/chacha-shared/weevil_vespula/test \
  --valid_triplets=/net/scratch/tianh/explain_teach/data/wv_triplets/w1=10e5_w2=1/valid_triplets.pkl \
  --test_triplets=/net/scratch/tianh/explain_teach/data/wv_triplets/w1=10e5_w2=1/test_triplets.pkl \
  --train_triplets=/net/scratch/tianh/explain_teach/data/wv_triplets/w1=10e5_w2=1/train_triplets.pkl \
  --num_class=2 \
  --embed_dim=10 \
  --max_epochs=100 \
  --learning_rate=1e-4 \
  --train_batch_size=120 \
  --transform=wv \
  --syn \
  --train_synthetic=/net/scratch/tianh/explain_teach/embeds/wv/train.pkl \
  --test_synthetic=/net/scratch/tianh/explain_teach/embeds/wv/test.pkl \
  --w1=10000 \
  --w2=1 \
  --lamda=0.8 \
  --do_train \
  --do_test

  
  # --train_triplets=/net/scratch/tianh/explain_teach/data/wv_triplets/w1=2.73_w2=1/train_triplets_25k.pkl \
  # --train_triplets=/net/scratch/chacha-shared/weevil_vespula/train_triplet.pkl \
  # --valid_triplets=/net/scratch/chacha-shared/weevil_vespula/valid_triplet.pkl \
  # --test_triplets=/net/scratch/chacha-shared/weevil_vespula/test_triplet.pkl \
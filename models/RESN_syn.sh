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
#SBATCH --nodelist=c001

python MTL.py \
  --wandb_mode=online \
  --wandb_entity=ai-driven-tutorial \
  --wandb_project=tests \
  --wandb_group=wv \
  --wandb_name=MTL \
  --train_dir=/net/scratch/chacha-shared/weevil_vespula/train \
  --valid_dir=/net/scratch/chacha-shared/weevil_vespula/valid \
  --test_dir=/net/scratch/chacha-shared/weevil_vespula/test \
  --train_triplets=/net/scratch/chacha-shared/weevil_vespula/train_triplet.pkl \
  --valid_triplets=/net/scratch/chacha-shared/weevil_vespula/valid_triplet.pkl \
  --test_triplets=/net/scratch/chacha-shared/weevil_vespula/test_triplet.pkl \
  --num_class=2 \
  --transform=wv \
  --embed_dim=10 \
  --max_epochs=50 \
  --train_batch_size=120 \
  --learning_rate=1e-4 \
  --syn \
  --train_synthetic=/net/scratch/tianh/explain_teach/embeds/wv/train.pkl \
  --test_synthetic=/net/scratch/tianh/explain_teach/embeds/wv/test.pkl \
  --w1=2.73027025 \
  --w2=1 \
  --do_train \
  --do_test
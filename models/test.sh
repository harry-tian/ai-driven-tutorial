#!/bin/bash
#
#SBATCH --mail-user=chacha@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/chacha/slurm/out/%j.%N.stdout
#SBATCH --error=/home/chacha/slurm/stderr/%j.%N.stderr
#SBATCH --job-name=triplets
#SBATCH --partition=cdac-contrib
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=20000
#SBATCH --nodelist=c001

python test.py \
  --wandb_mode=online \
  --wandb_project=wv \
  --wandb_group=w1=10000 \
  --wandb_name=RESN_test \
  --train_dir=/net/scratch/chacha-shared/weevil_vespula/train \
  --valid_dir=/net/scratch/chacha-shared/weevil_vespula/valid \
  --test_dir=/net/scratch/chacha-shared/weevil_vespula/test \
  --train_triplets=/net/scratch/tianh/explain_teach/data/wv_triplets/w1=10000_w2=1/train_triplets_10000.pkl \
  --valid_triplets=/net/scratch/tianh/explain_teach/data/wv_triplets/w1=10000_w2=1/valid_triplets_10000.pkl \
  --test_triplets=/net/scratch/tianh/explain_teach/data/wv_triplets/w1=10000_w2=1/test_triplets_10k.pkl \
  --num_class=2 \
  --embed_dim=10 \
  --transform=wv \
  --syn \
  --train_synthetic=/net/scratch/tianh/explain_teach/embeds/wv/train.pkl \
  --test_synthetic=/net/scratch/tianh/explain_teach/embeds/wv/test.pkl \
  --w1=10000 \
  --w2=1 \
  --ckpt_path=results/wv/3rk2widl/checkpoints/best_model.ckpt \

python test.py \
  --wandb_mode=online \
  --wandb_project=wv \
  --wandb_group=w1=10000 \
  --wandb_name=RESN_pretrained_test \
  --train_dir=/net/scratch/chacha-shared/weevil_vespula/train \
  --valid_dir=/net/scratch/chacha-shared/weevil_vespula/valid \
  --test_dir=/net/scratch/chacha-shared/weevil_vespula/test \
  --train_triplets=/net/scratch/tianh/explain_teach/data/wv_triplets/w1=10000_w2=1/train_triplets_10000.pkl \
  --valid_triplets=/net/scratch/tianh/explain_teach/data/wv_triplets/w1=10000_w2=1/valid_triplets_10000.pkl \
  --test_triplets=/net/scratch/tianh/explain_teach/data/wv_triplets/w1=10000_w2=1/test_triplets_10k.pkl \
  --num_class=2 \
  --embed_dim=10 \
  --transform=wv \
  --syn \
  --train_synthetic=/net/scratch/tianh/explain_teach/embeds/wv/train.pkl \
  --test_synthetic=/net/scratch/tianh/explain_teach/embeds/wv/test.pkl \
  --w1=10000 \
  --w2=1 \
  --ckpt_path=results/wv/2k25qfe5/checkpoints/best_model.ckpt \

python test.py \
  --wandb_mode=online \
  --wandb_project=wv \
  --wandb_group=w1=2.7303 \
  --wandb_name=RESN_test \
  --train_dir=/net/scratch/chacha-shared/weevil_vespula/train \
  --valid_dir=/net/scratch/chacha-shared/weevil_vespula/valid \
  --test_dir=/net/scratch/chacha-shared/weevil_vespula/test \
  --train_triplets=/net/scratch/tianh/explain_teach/data/wv_triplets/w1=2.7303_w2=1/train_triplets_10000.pkl \
  --valid_triplets=/net/scratch/tianh/explain_teach/data/wv_triplets/w1=2.7303_w2=1/valid_triplets_10000.pkl \
  --test_triplets=/net/scratch/tianh/explain_teach/data/wv_triplets/w1=2.7303_w2=1/test_triplets_10k.pkl \
  --num_class=2 \
  --embed_dim=10 \
  --transform=wv \
  --syn \
  --train_synthetic=/net/scratch/tianh/explain_teach/embeds/wv/train.pkl \
  --test_synthetic=/net/scratch/tianh/explain_teach/embeds/wv/test.pkl \
  --w1=2.7303 \
  --w2=1 \
  --ckpt_path=results/wv/kui29v1y/checkpoints/best_model.ckpt \

python test.py \
  --wandb_mode=online \
  --wandb_project=wv \
  --wandb_group=w1=2.7303 \
  --wandb_name=RESN_pretrained_test \
  --train_dir=/net/scratch/chacha-shared/weevil_vespula/train \
  --valid_dir=/net/scratch/chacha-shared/weevil_vespula/valid \
  --test_dir=/net/scratch/chacha-shared/weevil_vespula/test \
  --train_triplets=/net/scratch/tianh/explain_teach/data/wv_triplets/w1=2.7303_w2=1/train_triplets_10000.pkl \
  --valid_triplets=/net/scratch/tianh/explain_teach/data/wv_triplets/w1=2.7303_w2=1/valid_triplets_10000.pkl \
  --test_triplets=/net/scratch/tianh/explain_teach/data/wv_triplets/w1=2.7303_w2=1/test_triplets_10k.pkl \
  --num_class=2 \
  --embed_dim=10 \
  --transform=wv \
  --syn \
  --train_synthetic=/net/scratch/tianh/explain_teach/embeds/wv/train.pkl \
  --test_synthetic=/net/scratch/tianh/explain_teach/embeds/wv/test.pkl \
  --w1=2.7303 \
  --w2=1 \
  --ckpt_path=results/wv/11b1h2hh/checkpoints/best_model.ckpt \
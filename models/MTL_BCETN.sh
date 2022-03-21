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

hostname
echo $CUDA_VISIBLE_DEVICES

python MTL_BCETN.py \
  --embed_dim=10 \
  --wandb_entity=ai-driven-tutorial \
  --wandb_project=bm-htriplets \
  --wandb_group=MTL_lambdas \
  --wandb_mode=online \
  --train_triplets=/net/scratch/tianh/explain_teach/data/bm_triplets/3c2_unique=182/train_triplets.pkl \
  --valid_triplets=/net/scratch/tianh/explain_teach/data/bm_triplets/3c2_unique=182/valid_triplets.pkl \
  --pretrained \
  --max_epochs=200 \
  --learning_rate=1e-4 \
  --train_batch_size=64 \
  --do_train \
  --do_test \
  --seed=42 \
  --wandb_name=lambda=0.6 \
  --lamda=0.6 


python MTL_proto.py \
  --embed_dim=10 \
  --wandb_entity=ai-driven-tutorial \
  --wandb_project=bm-htriplets \
  --wandb_group=MTL_proto \
  --wandb_mode=online \
  --train_triplets=/net/scratch/tianh/explain_teach/data/bm_triplets/3c2_unique=182/train_triplets.pkl \
  --valid_triplets=/net/scratch/tianh/explain_teach/data/bm_triplets/3c2_unique=182/valid_triplets.pkl \
  --pretrained \
  --max_epochs=200 \
  --learning_rate=1e-4 \
  --train_batch_size=160 \
  --do_train \
  --lamda=0.2 \
  --ckpt=results/bm-htriplets/3kpi0afw/checkpoints/best_model.ckpt \
  --m=10
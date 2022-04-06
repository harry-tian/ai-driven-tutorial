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

   
python resn_args.py \
  --wandb_mode=online \
  --wandb_entity=ai-driven-tutorial \
  --wandb_project=nih \
  --wandb_group=10classes \
  --wandb_name=test \
  --train_dir=/net/scratch/tianh-shared/NIH/pathologies/train \
  --valid_dir=/net/scratch/tianh-shared/NIH/pathologies/valid \
  --test_dir=/net/scratch/tianh-shared/NIH/pathologies/valid \
  --num_class=10 \
  --embed_dim=10 \
  --pretrained \
  --max_epochs=200 \
  --train_batch_size=128 \
  --learning_rate=1e-4 \
  --transform=xray \
  --do_train \
  --do_test
  
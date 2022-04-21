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
#SBATCH --exclude=a[001-004,006,008],aa[001-003],d[001-002]


python MTL.py \
  --model_config=/net/scratch/tianh/explain_teach/models/configs/wv_2d/model.yaml \
  --dataset_config=/net/scratch/tianh/explain_teach/models/configs/wv_2d/dataset.yaml \
  --triplet_config=/net/scratch/tianh/explain_teach/models/configs/wv_2d/w1=2.7303.yaml \

python MTL.py \
  --model_config=/net/scratch/tianh/explain_teach/models/configs/wv_2d/model_pretrained.yaml \
  --dataset_config=/net/scratch/tianh/explain_teach/models/configs/wv_2d/dataset.yaml \
  --triplet_config=/net/scratch/tianh/explain_teach/models/configs/wv_2d/w1=2.7303.yaml \


  # --wandb_mode=online \
  # --wandb_project=wv \
  # --wandb_group=test \
  # --wandb_name=RESN \
  # --train_dir=/net/scratch/tianh-shared/wv-3d/pseudo_label/auto_split/train \
  # --valid_dir=/net/scratch/tianh-shared/wv-3d/pseudo_label/auto_split/valid \
  # --test_dir=/net/scratch/tianh-shared/wv-3d/pseudo_label/auto_split/test \
  # --train_triplets=/net/scratch/tianh/explain_teach/data/wv_triplets/w1=10000_w2=1/train_triplets_10000.pkl \
  # --valid_triplets=/net/scratch/tianh/explain_teach/data/wv_triplets/w1=10000_w2=1/valid_triplets_10000.pkl \
  # --test_triplets=/net/scratch/tianh/explain_teach/data/wv_triplets/w1=10000_w2=1/test_triplets_10k.pkl \
  # --num_class=2 \
  # --embed_dim=10 \
  # --max_epochs=50 \
  # --learning_rate=1e-4 \
  # --train_batch_size=98 \
  # --transform=wv \
  # --syn \
  # --train_synthetic=/net/scratch/tianh/explain_teach/embeds/wv/train.pkl \
  # --test_synthetic=/net/scratch/tianh/explain_teach/embeds/wv/test.pkl \
  # --w1=10000 \
  # --w2=1 \
  # --lamda=0.5 \
  # --do_train \
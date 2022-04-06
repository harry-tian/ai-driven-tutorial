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
#SBATCH --nodelist=aa[001]

hostname
echo $CUDA_VISIBLE_DEVICES

# export PATH=$PATH:/usr/local/cuda/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH=/usr/local/cuda/lib
# export FOLD=$SLURM_ARRAY_TASK_ID
# export FOLD=4

# nvidia-smi -l 1 &

# /home/hanliu/anaconda3/bin/python dwac_args.py \
#   --mri_sequences=RGB \
#   --data_sequences=RGB \
#   --embed_dim=10 \
#   --merge_dim=2 \
#   --merge_seq \
#   --pooling \
#   --wandb_mode=online \
#   --wandb_group=dwac-emb10-mrg2 \
#   --output_dir=results/dwac-emb10-mrg2 \
#   --train_dir=/net/scratch/hanliu/radiology/explain_teach/data/bm/train \
#   --valid_dir=/net/scratch/hanliu/radiology/explain_teach/data/bm/valid \
#   --max_epochs=100 \
#   --learning_rate=1e-4 \
#   --train_batch_size=16 \
#   --do_train

  # --horizontal_flip=0.5 \

python dwac_args.py \
  --mri_sequences=RGB \
  --data_sequences=RGB \
  --embed_dim=10 \
  --merge_dim=10 \
  --merge_seq \
  --pooling \
  --wandb_mode=online \
  --wandb_entity=ai-driven-tutorial \
  --wandb_project=butterfly-moth \
  --wandb_group=dwac-emb10-mrg10 \
  --train_dir=/net/scratch/tianh/bm/train \
  --valid_dir=/net/scratch/tianh/bm/valid \
  --max_epochs=200 \
  --learning_rate=1e-4 \
  --train_batch_size=16 \
  --do_train
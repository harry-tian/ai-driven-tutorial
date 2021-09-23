#!/bin/bash
#
#SBATCH --output=/home/hanliu/slurm_out/%j.%N.stdout
#SBATCH --error=/home/hanliu/slurm_out/%j.%N.stderr
#SBATCH --job-name=train
#SBATCH --partition=dev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=20000
#SBATCH --chdir=/net/scratch/hanliu/radiology/explain_teach/model
# #SBATCH --array=0-4

hostname
echo $CUDA_VISIBLE_DEVICES

# export PATH=$PATH:/usr/local/cuda/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH=/usr/local/cuda/lib
# export FOLD=$SLURM_ARRAY_TASK_ID
# export FOLD=4

# nvidia-smi -l 1 &

/home/hanliu/anaconda3/bin/python dres_args.py \
  --embed_dim=10 \
  --wandb_mode=online \
  --wandb_group=dres-emb10 \
  --output_dir=results/dres-emb10 \
  --train_dir=/net/scratch/hanliu/radiology/explain_teach/data/bm/train \
  --valid_dir=/net/scratch/hanliu/radiology/explain_teach/data/bm/valid \
  --dataloader_num_workers=4 \
  --gpus=1 \
  --seed=42 \
  --max_epochs=100 \
  --learning_rate=1e-4 \
  --vertical_flip=0.5 \
  --rotate=30 \
  --scale=0.2 \
  --train_batch_size=16 \
  --do_train

  # --horizontal_flip=0.5 \

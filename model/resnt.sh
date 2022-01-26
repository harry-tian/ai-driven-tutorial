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

/home/hanliu/anaconda3/bin/python resnt_args.py \
  --embed_dim=10 \
  --wandb_mode=online \
  --wandb_group=resnt-emb10-200 \
  --output_dir=results/resnt-emb10 \
  --train_dir=/net/scratch/hanliu/radiology/explain_teach/data/bm/train \
  --valid_dir=/net/scratch/hanliu/radiology/explain_teach/data/bm/valid \
  --train_pairwise_distance=/net/scratch/hanliu/radiology/explain_teach/lpips.bm.train.pkl \
  --valid_pairwise_distance=/net/scratch/hanliu/radiology/explain_teach/lpips.bm.valid.pkl \
  --dataloader_num_workers=4 \
  --gpus=1 \
  --seed=42 \
  --max_epochs=200 \
  --learning_rate=1e-4 \
  --vertical_flip=0.5 \
  --rotate=30 \
  --scale=0.2 \
  --train_batch_size=16 \
  --do_train

  # --horizontal_flip=0.5 \

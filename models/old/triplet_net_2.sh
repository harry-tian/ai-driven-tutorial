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
#SBATCH --nodelist=aa[002-003]

hostname
echo $CUDA_VISIBLE_DEVICES

# export PATH=$PATH:/usr/local/cuda/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH=/usr/local/cuda/lib
# export FOLD=$SLURM_ARRAY_TASK_ID
# export FOLD=4

# nvidia-smi -l 1 &

python triplet_net_2_args.py \
  --wandb_group=triplet_net_2 \
  --embed_dim=10 \
  --max_epochs=10 \
  --learning_rate=1e-6 \
  --train_batch_size=64 \
  --output_dir=results/triplet \
  --train_dir=/net/scratch/hanliu-shared/data/bm/train \
  --valid_dir=/net/scratch/hanliu-shared/data/bm/valid \
  --train_pairwise_distance=../embeds/lpips.bm.train.pkl \
  --valid_pairwise_distance=../embeds/lpips.bm.valid.pkl \
  --do_train

  # --horizontal_flip=0.5 \

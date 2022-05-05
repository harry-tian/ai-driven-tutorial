#!/bin/bash
#
#SBATCH --mail-user=tianh@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/tianh/slurm/out/%j.%N.stdout
#SBATCH --error=/home/tianh/slurm/stderr/%j.%N.stderr
#SBATCH --job-name=triplets
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
  

python RESN_test.py \
    --model_config=/net/scratch/tianh/explain_teach/models/configs/wv_2d/RESN.yaml \
    --dataset_config=/net/scratch/tianh/explain_teach/models/configs/wv_2d/dataset.yaml \
    --triplet_config=/net/scratch/tianh/explain_teach/models/configs/wv_2d/w1=1_w2=0.yaml \
  

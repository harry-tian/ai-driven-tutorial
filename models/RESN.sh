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
  
python RESN.py \
  --model_config=/net/scratch/tianh/explain_teach/models/configs/wv_3d/model_pretrained.yaml \
  --dataset_config=/net/scratch/tianh/explain_teach/models/configs/wv_3d/dataset.yaml \
  --triplet_config=/net/scratch/tianh/explain_teach/models/configs/wv_3d/align=1.yaml \
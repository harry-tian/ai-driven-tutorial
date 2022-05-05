#!/bin/bash
#
#SBATCH --mail-user=tianh@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/tianh/slurm/out/%j.%N.stdout
#SBATCH --error=/home/tianh/slurm/stderr/%j.%N.stderr
#SBATCH --job-name=TN_syn
#SBATCH --partition=cdac-contrib
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=50Gs
#SBATCH --exclude=aa[001-002]


# python RESN.py \
#     --model_config=/net/scratch/tianh/explain_teach/models/configs/bm/RESN.yaml \
#     --dataset_config=/net/scratch/tianh/explain_teach/models/configs/bm/dataset.yaml \
#     --triplet_config=/net/scratch/tianh/explain_teach/models/configs/bm/triplets.yaml \

# python TN.py \
#     --model_config=/net/scratch/tianh/explain_teach/models/configs/bm/TN.yaml \
#     --dataset_config=/net/scratch/tianh/explain_teach/models/configs/bm/dataset.yaml \
#     --triplet_config=/net/scratch/tianh/explain_teach/models/configs/bm/triplets.yaml \

python MTL.py \
    --model_config=/net/scratch/tianh/explain_teach/models/configs/bm/MTL.yaml \
    --dataset_config=/net/scratch/tianh/explain_teach/models/configs/bm/dataset.yaml \
    --triplet_config=/net/scratch/tianh/explain_teach/models/configs/bm/triplets.yaml \

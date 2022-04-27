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
#SBATCH --nodelist=c001

python MTL.py \
    --model_config=/net/scratch/tianh/explain_teach/models/configs/wv_3d/align=0.3.yaml \
    --triplet_config=/net/scratch/tianh/explain_teach/models/configs/wv_3d/MTL0.5.yaml \

python MTL.py \
    --model_config=/net/scratch/tianh/explain_teach/models/configs/wv_3d/align=0.2.yaml \
    --triplet_config=/net/scratch/tianh/explain_teach/models/configs/wv_3d/MTL0.5.yaml \

python MTL.py \
    --model_config=/net/scratch/tianh/explain_teach/models/configs/wv_3d/align=0.1.yaml \
    --triplet_config=/net/scratch/tianh/explain_teach/models/configs/wv_3d/MTL0.5.yaml \

python MTL.py \
    --model_config=/net/scratch/tianh/explain_teach/models/configs/wv_3d/align=0.3.yaml \
    --triplet_config=/net/scratch/tianh/explain_teach/models/configs/wv_3d/MTL0.8.yaml \

python MTL.py \
    --model_config=/net/scratch/tianh/explain_teach/models/configs/wv_3d/align=0.2.yaml \
    --triplet_config=/net/scratch/tianh/explain_teach/models/configs/wv_3d/MTL0.8.yaml \

python MTL.py \
    --model_config=/net/scratch/tianh/explain_teach/models/configs/wv_3d/align=0.1.yaml \
    --triplet_config=/net/scratch/tianh/explain_teach/models/configs/wv_3d/MTL0.8.yaml \

python MTL.py \
    --model_config=/net/scratch/tianh/explain_teach/models/configs/wv_3d/align=0.0.yaml \
    --triplet_config=/net/scratch/tianh/explain_teach/models/configs/wv_3d/MTL0.8.yaml \

python MTL.py \
    --model_config=/net/scratch/tianh/explain_teach/models/configs/wv_3d/align=0.0.yaml \
    --triplet_config=/net/scratch/tianh/explain_teach/models/configs/wv_3d/MTL0.5.yaml \
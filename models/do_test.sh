#!/bin/bash
#
#SBATCH --mail-user=tianh@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/tianh/slurm/out/%j.%N.stdout
#SBATCH --error=/home/tianh/slurm/stderr/%j.%N.stderr
#SBATCH --job-name=triplets
#SBATCH --partition=dev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=10G

# python do_test.py \
#     --dataset_config=configs/wv_2d/dataset.yaml \
#     --model_config=configs/wv_2d/models/RESN.yaml \
#     --triplet_config=configs/wv_2d/triplets/align=1.yaml \
#     --ckpt_path=checkpoints/wv_2d/2s2hr40c/checkpoints/best_model.ckpt \
#     --seed=0 

for file in configs/wv_2d/triplets/* 
    do if [ $file != configs/wv_2d/triplets/RESN.yaml ]
        then python do_test.py \
                --dataset_config=configs/wv_2d/dataset.yaml \
                --model_config=configs/wv_2d/models/RESN.yaml \
                --triplet_config=$file \
                --ckpt_path=checkpoints/wv_2d/2s2hr40c/checkpoints/best_model.ckpt \
                --seed=0 
    fi
done

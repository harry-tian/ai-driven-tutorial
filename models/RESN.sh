#!/bin/bash
#
#SBATCH --mail-user=tianh@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/tianh/slurm/out/%j.%N.stdout
#SBATCH --error=/home/tianh/slurm/stderr/%j.%N.stderr
#SBATCH --job-name=triplets
#SBATCH --partition=cdac-own
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=10G

for i in {0..9}
    do  python MTL_han.py \
            --dataset_config=configs/wv_2d/dataset.yaml \
            --model_config=configs/wv_2d/models/RESN.yaml \
            --triplet_config=configs/wv_2d/align_triplets/align=0.7.yaml \
            --overwrite_config=configs/wv_2d/overwrite.yaml \
            --seed=$i
done

# for file in configs/wv_2d/triplets/* ; do sbatch RESN.sh $file ; done
# python main.py \
#             --dataset_config=configs/wv_2d/dataset.yaml \
#             --model_config=configs/wv_2d/models/RESN.yaml \
#             --triplet_config=configs/wv_2d/triplets/align=0.8.yaml \
#             --seed=0
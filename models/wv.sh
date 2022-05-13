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
#SBATCH --exclude=aa[001-002]


for seed in {0..2}
    do python MTL.py \
        --dataset_config=configs/wv_3d/dataset.yaml \
        --model_config=$2 \
        --triplet_config=$1 \
        --overwrite_config=configs/wv_3d/overwrite.yaml \
        --seed=$seed
done

# for triplet in configs/wv_3d/noisy_triplets/* ; do for model in configs/wv_3d/models/* ; do if [ $model != configs/wv_3d/models/RESN.yaml ]; then sbatch wv.sh $triplet $model ; fi; done; done
 

# python MTL.py \
#     --dataset_config=configs/wv_3d/dataset.yaml \
#     --model_config=$2 \
#     --triplet_config=$1 \
#     --overwrite_config=configs/wv_3d/overwrite.yaml \
#     --seed=$3

# for triplet in configs/wv_3d/align_triplets/* ; do for model in configs/wv_3d/models/* ; do if [ $model != configs/wv_3d/models/RESN.yaml ]; then sbatch wv.sh $triplet $model 2; fi; done; done

# for triplet in configs/wv_3d/filtered_triplets/* ; do for model in configs/wv_3d/models/* ; do if [ $model != configs/wv_3d/models/RESN.yaml ]; then sbatch wv.sh $triplet $model 2; fi; done; done



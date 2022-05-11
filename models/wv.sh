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

 
# python MTL_han.py \
#     --dataset_config=configs/wv_2d/dataset.yaml \
#     --model_config=configs/wv_2d/models/MTL0.5.yaml \
#     --triplet_config=configs/wv_2d/align_triplets/align=0.7.yaml \
#     --overwrite_config=configs/wv_2d/overwrite.yaml \
#     --seed=0
#  

# for triplet in configs/wv_2d/filtered_triplets/* ; do for model in configs/wv_2d/models/* ; do if [ $model = configs/wv_2d/models/TN.yaml ] || [ $model = configs/wv_2d/models/MTL0.5.yaml ]; then sbatch wv.sh $triplet $model; fi; done; done

# for triplet in configs/wv_2d/align_triplets/* ; do for model in configs/wv_2d/models/* ; do if [ $model = configs/wv_2d/models/TN.yaml ] || [ $model = configs/wv_2d/models/MTL0.5.yaml ]; then sbatch wv.sh $triplet $model; fi; done; done

for i in {0..2}
    # do echo $1 $2 $i
    do python MTL_han.py \
        --dataset_config=configs/wv_2d/dataset.yaml \
        --model_config=$2 \
        --triplet_config=$1 \
        --seed=$i
done



        # --overwrite_config=configs/wv_2d/overwrite.yaml \
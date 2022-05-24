#!/bin/bash
#
#SBATCH --mail-user=tianh@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/tianh/slurm/out/%j.%N.stdout
#SBATCH --error=/home/tianh/slurm/stderr/%j.%N.stderr
#SBATCH --job-name=triplets
#SBATCH --partition=cdac-contrib
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --exclude=aa[001-002]

# python main.py \
#                 --dataset_config=configs/wv_3d_square/dataset.yaml \
#                 --model_config=configs/models/MTL0.5.yaml \
#                 --triplet_config=configs/wv_3d_square/triplets/unfiltered/noisy_0.925/p=0.1.yaml \
#                 --overwrite_config=configs/wv_3d_square/overwrite.yaml \
#                 --seed=0

for seed in {0..2}
        do      python main.py \
                --dataset_config=configs/wv_3d_square/dataset.yaml \
                --model_config=$2 \
                --triplet_config=$1 \
                --overwrite_config=configs/wv_3d_square/overwrite.yaml \
                --seed=$seed
done

# for triplet in configs/wv_3d_square/triplets/unfiltered/aligns/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv_square.sh $triplet $model ; fi; done; done

# for triplet in configs/wv_3d_square/triplets/filtered/aligns/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv_square.sh $triplet $model ; fi; done; done

# for triplet in configs/wv_3d_square/triplets/unfiltered/noisy_0.925/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv_square.sh $triplet $model ; fi; done; done

# for triplet in configs/wv_3d_square/triplets/filtered/noisy_0.925/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv_square.sh $triplet $model ; fi; done; done

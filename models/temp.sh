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

python MTL_slow.py \
        --dataset_config=configs/wv_3d/square.yaml \
        --model_config=configs/wv_3d/models/TN.yaml \
        --triplet_config=$1 \
        --overwrite_config=configs/wv_3d/square/num.yaml \
        --seed=0

# for triplet in configs/wv_3d/square/num_0.925/* ; do sbatch temp.sh $triplet ;done

# for triplet in configs/wv_3d/square/filtered/* ; do sbatch temp.sh $triplet ;done

# for triplet in configs/wv_3d/square/num_0.925/* ; do for model in configs/wv_3d/models/* ; do if [ $model = configs/wv_3d/models/TN.yaml ] ; then sbatch temp.sh $triplet $model $seed; fi; done; done
# for triplet in configs/wv_3d/square/num_0.8/* ; do for model in configs/wv_3d/models/* ; do if [ $model = configs/wv_3d/models/TN.yaml ] ; then sbatch temp.sh $triplet $model $seed; fi; done; done

# for seed in {0..2}; do for triplet in configs/wv_3d/square/num_0.925/* ; do for model in configs/wv_3d/models/* ; do if [ $model = configs/wv_3d/models/TN.yaml ] || [ $model = configs/wv_3d/models/MTL0.5.yaml ]; then sbatch temp.sh $triplet $model $seed; fi; done; done; done
# for seed in {0..2}; do for triplet in configs/wv_3d/square/num_0.8/* ; do for model in configs/wv_3d/models/* ; do if [ $model = configs/wv_3d/models/TN.yaml ] || [ $model = configs/wv_3d/models/MTL0.5.yaml ]; then sbatch wv.sh $triplet $model $seed; fi; done; done; done





# python gen_RESN.py \
#     --dataset=bm \
#     --subdir=bm/RESN_baseline \
#     --suffix=emb512


# for i in {0..4}
#       do python MTL.py \
#       --dataset_config=configs/wv_3d/dataset.yaml \
#       --model_config=configs/wv_3d/models/RESN.yaml \
#       --triplet_config=configs/wv_3d/align_triplets/align=0.5.yaml \
#       --overwrite_config=configs/wv_3d/overwrite.yaml \
#       --seed=$i
# done

    
# python MTL.py \
#     --dataset_config=configs/wv_3d/dataset.yaml \
#     --model_config=configs/wv_3d/models/TN.yaml \
#     --triplet_config=configs/wv_3d/align_triplets/align=0.8.yaml \
#     --overwrite_config=configs/wv_3d/overwrite.yaml \
#     --seed=0

# python MTL.py \
#     --dataset_config=configs/wv_3d/dataset.yaml \
#     --model_config=configs/wv_3d/models/MTL0.5.yaml \
#     --triplet_config=configs/wv_3d/filtered_triplets/align=0.8_filtered.yaml \
#     --overwrite_config=configs/wv_3d/overwrite.yaml \
#     --seed=0

# python MTL.py \
#     --dataset_config=configs/wv_3d/dataset.yaml \
#     --model_config=configs/wv_3d/models/MTL0.5.yaml \
#     --triplet_config=configs/wv_3d/noisy_triplets/noise=0.1.yaml \
#     --overwrite_config=configs/wv_3d/overwrite.yaml \
#     --seed=0
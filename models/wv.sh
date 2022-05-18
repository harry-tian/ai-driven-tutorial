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

python MTL_slow.py \
        --dataset_config=configs/wv_3d/square.yaml \
        --model_config=$2 \
        --triplet_config=$1 \
        --overwrite_config=configs/wv_3d/overwrite.yaml \
        --seed=$3

# for seed in {0..2}; do for triplet in configs/wv_3d/square/num_0.925/* ; do for model in configs/wv_3d/models/* ; do if [ $model = configs/wv_3d/models/TN.yaml ] || [ $model = configs/wv_3d/models/MTL0.5.yaml ]; then sbatch temp.sh $triplet $model $seed; fi; done; done; done
# for seed in {0..2}; do for triplet in configs/wv_3d/square/num_0.8/* ; do for model in configs/wv_3d/models/* ; do if [ $model = configs/wv_3d/models/TN.yaml ] || [ $model = configs/wv_3d/models/MTL0.5.yaml ]; then sbatch wv.sh $triplet $model $seed; fi; done; done; done



# for seed in {0..2}; do for triplet in configs/wv_3d/square/noisy_0.925/* ; do for model in configs/wv_3d/models/* ; do if [ $model = configs/wv_3d/models/TN.yaml ] || [ $model = configs/wv_3d/models/MTL0.5.yaml ]; then sbatch wv.sh $triplet $model $seed; fi; done; done; done
# for seed in {0..2}; do for triplet in configs/wv_3d/square/noisy_0.8/* ; do for model in configs/wv_3d/models/* ; do if [ $model = configs/wv_3d/models/TN.yaml ] || [ $model = configs/wv_3d/models/MTL0.5.yaml ]; then sbatch wv.sh $triplet $model $seed; fi; done; done; done


# for triplet in configs/wv_3d/square/num_0.925/* ; do for model in configs/wv_3d/models/* ; do if [ $model != configs/wv_3d/models/RESN.yaml ]; then sbatch wv.sh $triplet $model ; fi; done; done
# for triplet in configs/wv_3d/square/num_0.8/* ; do for model in configs/wv_3d/models/* ; do if [ $model != configs/wv_3d/models/RESN.yaml ]; then sbatch wv.sh $triplet $model ; fi; done; done


# for triplet in configs/wv_3d/triplets/filtered/* ; do for model in configs/wv_3d/models/* ; do if [ $model != configs/wv_3d/models/RESN.yaml ]; then sbatch wv.sh $triplet $model ; fi; done; done
# for triplet in configs/wv_3d/triplets/unfiltered/* ; do for model in configs/wv_3d/models/* ; do if [ $model != configs/wv_3d/models/RESN.yaml ]; then sbatch wv.sh $triplet $model ; fi; done; done



# for triplet in configs/wv_3d/triplets/num_0.63_filtered/* ; do for model in configs/wv_3d/models/* ; do if [ $model != configs/wv_3d/models/RESN.yaml ]; then sbatch wv.sh $triplet $model ; fi; done; done
# for triplet in configs/wv_3d/triplets/num_0.8_filtered/* ; do for model in configs/wv_3d/models/* ; do if [ $model != configs/wv_3d/models/RESN.yaml ]; then sbatch wv.sh $triplet $model ; fi; done; done
# for triplet in configs/wv_3d/triplets/noisy_0.63_filtered/* ; do for model in configs/wv_3d/models/* ; do if [ $model != configs/wv_3d/models/RESN.yaml ]; then sbatch wv.sh $triplet $model ; fi; done; done
# for triplet in configs/wv_3d/triplets/noisy_0.8_filtered/* ; do for model in configs/wv_3d/models/* ; do if [ $model != configs/wv_3d/models/RESN.yaml ]; then sbatch wv.sh $triplet $model ; fi; done; done



# python MTL.py \
#         --dataset_config=configs/wv_3d/triplet.yaml \
#         --model_config=configs/wv_3d/models/MTL0.5.yaml \
#         --triplet_config=configs/wv_3d/triplets/num_0.63/p=0.125.yaml \
#         --overwrite_config=configs/wv_3d/overwrite.yaml \
#         --seed=0




 



# for triplet in configs/wv_3d/filtered_triplets/* ; do for model in configs/wv_3d/models/* ; do if [ $model != configs/wv_3d/models/RESN.yaml ]; then sbatch wv.sh $triplet $model 0; fi; done; done



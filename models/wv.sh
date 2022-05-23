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
#         --dataset_config=configs/wv_3d_linear/dataset.yaml \
#         --model_config=configs/wv_3d_linear/models/MTL0.2.yaml \
#         --triplet_config=configs/wv_3d_linear/triplets/num_0.97/p=0.0078125.yaml \
#         --overwrite_config=configs/wv_3d_linear/overwrite_num.yaml \
#         --seed=0 \

for seed in {0..2}
        do      python main.py \
                --dataset_config=configs/wv_3d_linear/dataset.yaml \
                --model_config=$2 \
                --triplet_config=$1 \
                --overwrite_config=configs/wv_3d_linear/overwrite_num.yaml \
                --seed=$seed
done

# for triplet in configs/wv_3d_linear/triplets/num_0.62/* ; do for model in configs/wv_3d_linear/models/* ; do if [ $model != configs/wv_3d_linear/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model ; fi; done; done
# for triplet in configs/wv_3d_linear/triplets/num_0.97/* ; do for model in configs/wv_3d_linear/models/* ; do if [ $model != configs/wv_3d_linear/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model ; fi; done; done


# for triplet in configs/wv_3d_linear/triplets/aligns/* ; do for model in configs/wv_3d_linear/models/* ; do if [ $model != configs/wv_3d_linear/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model ; fi; done; done
# for triplet in configs/wv_3d_linear/triplets/noisy_0.62/* ; do for model in configs/wv_3d_linear/models/* ; do if [ $model != configs/wv_3d_linear/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model ; fi; done; done
# for triplet in configs/wv_3d_linear/triplets/noisy_0.97/* ; do for model in configs/wv_3d_linear/models/* ; do if [ $model != configs/wv_3d_linear/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model ; fi; done; done


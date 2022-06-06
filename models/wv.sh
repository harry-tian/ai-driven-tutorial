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

DATA=wv_linear_sm


###### for CDAC-CONTRIB ###########

DIMS=(50 512)
overwrite=("filtered" "unfiltered")
for i in {0..1}
    do for j in {0..1}
        do  python main.py \
            --dataset_config=configs/$DATA/dataset.yaml \
            --model_config=$2 \
            --triplet_config=$1 \
            --overwrite_config=configs/"${overwrite[j]}".yaml \
            --seed=$3 \
            --embed_dim="${DIMS[i]}" \
    done
done


# DATA=wv_linear_sm; for seed in {0..2}; do for triplet in configs/$DATA/triplets/aligns/* ; do for model in configs/models/* ; do if [ $model == configs/models/MTL0.5.yaml ] ; then sbatch wv.sh $triplet $model $seed; fi; done; done; done


# python main.py \
#     --dataset_config=configs/$DATA/dataset.yaml \
#     --model_config=configs/models/MTL0.5.yaml \
#     --triplet_config=configs/$DATA/triplets/aligns/align=0.94.yaml \
#     --overwrite_config=configs/unfiltered.yaml \
#     --seed=0 \
#     --embed_dim=512 

###### for DEV ###########

# python main.py \
#     --dataset_config=configs/$DATA/dataset.yaml \
#     --model_config=$2 \
#     --triplet_config=$1 \
#     --overwrite_config=configs/overwrite.yaml \
#     --seed=$3 \
#     --embed_dim=$4 \
#     --max_epochs=10

# DATA=wv_linear_sm; DIMS=(50 512); for seed in {0..2} ; do for i in {0..1}; do for triplet in configs/$DATA/triplets/filtered/aligns/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model $seed "${DIMS[i]}"; fi; done; done; done; done




# DATA=wv_linear_sm; for triplet in configs/$DATA/triplets/unfiltered/noisy_0.84/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model ; fi; done; done
# DATA=wv_linear_sm; for triplet in configs/$DATA/triplets/filtered/noisy_0.84/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model ; fi; done; done
# DATA=wv_linear_sm; for triplet in configs/$DATA/triplets/unfiltered/num_0.84/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model ; fi; done; done
# DATA=wv_linear_sm; for triplet in configs/$DATA/triplets/filtered/num_0.84/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model ; fi; done; done



# DATA=wv_linear_sm; for triplet in configs/$DATA/triplets/filtered/aligns/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model; fi; done; done
# DATA=wv_linear_sm; for triplet in configs/$DATA/triplets/unfiltered/aligns/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model; fi; done; done









# python main.py \
#                 --dataset_config=configs/$DATA/dataset.yaml \
#                 --model_config=configs/models/MTL0.5.yaml \
#                 --triplet_config=configs/$DATA/triplets/unfiltered/aligns/align=0.9.yaml \
#                 --overwrite_config=configs/overwrite.yaml \
#                 --seed=0 \
#                 --embed_dim=50 \
#                 --max_epochs=1









# DATA=wv_linear_sm; for seed in {0..1} ;do for triplet in configs/$DATA/triplets/unfiltered/aligns/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model $seed; fi; done; done; done



# DATA=wv_linear_sm; for triplet in configs/$DATA/triplets/unfiltered/noisy_0.9/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model ; fi; done; done
# DATA=wv_linear_sm; for triplet in configs/$DATA/triplets/filtered/noisy_0.9/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model ; fi; done; done
# DATA=wv_linear_sm; for triplet in configs/$DATA/triplets/unfiltered/num_0.9/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model ; fi; done; done


# DATA=wv_linear_sm; for seed in {0..1} ; do for triplet in configs/$DATA/triplets/filtered/num_0.9/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model $seed; fi; done; done; done

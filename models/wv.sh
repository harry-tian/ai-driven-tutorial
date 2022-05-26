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
#SBATCH --exclude=aa[001-002]

DATA=wv_squarelin



python main.py \
    --dataset_config=configs/$DATA/dataset.yaml \
    --model_config=$2 \
    --triplet_config=$1 \
    --overwrite_config=configs/overwrite.yaml \
    --seed=$3 \
    --embed_dim=$4

# DATA=wv_squarelin; DIMS=(50 512); for seed in {0..1} ; do for i in {0..1}; do for triplet in configs/$DATA/triplets/unfiltered/aligns/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model $seed "${DIMS[i]}"; fi; done; done; done; done


# DIMS=(50 512)
# for i in {0..1}
#     do for seed in {0..2}
#             do      python main.py \
#                     --dataset_config=configs/$DATA/dataset.yaml \
#                     --model_config=$2 \
#                     --triplet_config=$1 \
#                     --overwrite_config=configs/overwrite.yaml \
#                     --seed=$seed \
#                     --embed_dim="${DIMS[i]}"
#     done
# done

# DATA=wv_squarelin; for triplet in configs/$DATA/triplets/filtered/aligns/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model; fi; done; done
# DATA=wv_squarelin; for triplet in configs/$DATA/triplets/unfiltered/aligns/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model; fi; done; done


# DATA=wv_squarelin; for triplet in configs/$DATA/triplets/unfiltered/noisy_0.8/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model ; fi; done; done
# DATA=wv_squarelin; for triplet in configs/$DATA/triplets/filtered/noisy_0.8/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model ; fi; done; done
# DATA=wv_squarelin; for triplet in configs/$DATA/triplets/unfiltered/num_0.8/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model ; fi; done; done
# DATA=wv_squarelin; for triplet in configs/$DATA/triplets/filtered/num_0.8/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model ; fi; done; done







# python main.py \
#                 --dataset_config=configs/$DATA/dataset.yaml \
#                 --model_config=configs/models/MTL0.5.yaml \
#                 --triplet_config=configs/$DATA/triplets/unfiltered/aligns/align=0.9.yaml \
#                 --overwrite_config=configs/overwrite.yaml \
#                 --seed=0 \
#                 --embed_dim=50 \
#                 --max_epochs=1









# DATA=wv_squarelin; for seed in {0..1} ;do for triplet in configs/$DATA/triplets/unfiltered/aligns/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model $seed; fi; done; done; done



# DATA=wv_squarelin; for triplet in configs/$DATA/triplets/unfiltered/noisy_0.9/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model ; fi; done; done
# DATA=wv_squarelin; for triplet in configs/$DATA/triplets/filtered/noisy_0.9/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model ; fi; done; done
# DATA=wv_squarelin; for triplet in configs/$DATA/triplets/unfiltered/num_0.9/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model ; fi; done; done


# DATA=wv_squarelin; for seed in {0..1} ; do for triplet in configs/$DATA/triplets/filtered/num_0.9/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model $seed; fi; done; done; done

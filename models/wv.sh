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

DATA=wv_3d_linear0

# python main.py \
#                 --dataset_config=configs/$DATA/dataset.yaml \
#                 --model_config=configs/models/MTL0.5.yaml \
#                 --triplet_config=configs/$DATA/triplets/unfiltered/aligns/align=1.yaml \
#                 --overwrite_config=configs/$DATA/overwrite.yaml \
#                 --seed=0 \
#                 --embed_dim=50

DIMS=(50 512)
for i in {0..1}
    do for seed in {0..2}
            do      python main.py \
                    --dataset_config=configs/$DATA/dataset.yaml \
                    --model_config=$2 \
                    --triplet_config=$1 \
                    --overwrite_config=configs/$DATA/overwrite.yaml \
                    --seed=$seed \
                    --embed_dim="${DIMS[i]}"
    done
done

# DATA=wv_3d_linear0; for triplet in configs/$DATA/triplets/unfiltered/aligns/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model ; fi; done; done

# DATA=wv_3d_linear0; for triplet in configs/$DATA/triplets/filtered/aligns/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model ; fi; done; done


# DATA=wv_3d_linear0; for triplet in configs/$DATA/triplets/unfiltered/noisy_0.9/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model ; fi; done; done
# DATA=wv_3d_linear0; for triplet in configs/$DATA/triplets/filtered/noisy_0.9/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model ; fi; done; done
# DATA=wv_3d_linear0; for triplet in configs/$DATA/triplets/unfiltered/num_0.9/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model ; fi; done; done
# DATA=wv_3d_linear0; for triplet in configs/$DATA/triplets/filtered/num_0.9/* ; do for model in configs/models/* ; do if [ $model != configs/models/RESN.yaml ] ; then sbatch wv.sh $triplet $model ; fi; done; done

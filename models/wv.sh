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
#SBATCH --exclude=a[001-002],a006
# for file in configs/wv_2d/models/* ; do for i in {0..4} ; do if [ $file != configs/wv_2d/models/RESN.yaml ]; then sbatch wv.sh $file $i; fi; done; done

python main.py \
    --dataset_config=configs/wv_2d/dataset.yaml \
    --model_config=$1 \
    --triplet_config=configs/wv_2d/triplets/w1=1_w2=0_filtered.yaml \
    --seed=$2

# for file in configs/wv_2d/models/* ; do 
#     for i in {3..4} ; do 
#         if [ $file = configs/wv_2d/models/MTL0.2.yaml ]
#         then python main.py \
#             --dataset_config=configs/wv_2d/dataset.yaml \
#             --model_config=$file \
#             --triplet_config=configs/wv_2d/triplets/w1=1_w2=0_filtered.yaml \
#             --seed=$i 
#         fi 
#     done
# done

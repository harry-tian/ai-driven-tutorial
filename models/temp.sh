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
    --model_config=configs/wv_2d/models/TN.yaml \
    --triplet_config=configs/wv_2d/triplets/align=0.7_filtered.yaml \
    --seed=0

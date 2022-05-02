#!/bin/bash
#
#SBATCH --mail-user=tianh@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/tianh/slurm/out/%j.%N.stdout
#SBATCH --error=/home/tianh/slurm/stderr/%j.%N.stderr
#SBATCH --job-name=sweep
#SBATCH --partition=cdac-contrib
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --exclude=aa[001-003]

wandb agent ai-driven-tutorial/bm_prolific/b5asiwq7
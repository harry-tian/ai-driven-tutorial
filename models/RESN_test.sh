#!/bin/bash
#
#SBATCH --mail-user=tianh@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/tianh/slurm/out/%j.%N.stdout
#SBATCH --error=/home/tianh/slurm/stderr/%j.%N.stderr
#SBATCH --job-name=triplets
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
  
python RESN_test.py --model_config=/net/scratch/tianh/explain_teach/models/configs/wv_3d/align=1.0.yaml 

python RESN_test.py --model_config=/net/scratch/tianh/explain_teach/models/configs/wv_3d/align=0.9.yaml 

python RESN_test.py --model_config=/net/scratch/tianh/explain_teach/models/configs/wv_3d/align=0.8.yaml 

python RESN_test.py --model_config=/net/scratch/tianh/explain_teach/models/configs/wv_3d/align=0.7.yaml 

python RESN_test.py --model_config=/net/scratch/tianh/explain_teach/models/configs/wv_3d/align=0.6.yaml 

python RESN_test.py --model_config=/net/scratch/tianh/explain_teach/models/configs/wv_3d/align=0.5.yaml 

python RESN_test.py --model_config=/net/scratch/tianh/explain_teach/models/configs/wv_3d/align=0.4.yaml 

python RESN_test.py --model_config=/net/scratch/tianh/explain_teach/models/configs/wv_3d/align=0.3.yaml 

python RESN_test.py --model_config=/net/scratch/tianh/explain_teach/models/configs/wv_3d/align=0.2.yaml 

python RESN_test.py --model_config=/net/scratch/tianh/explain_teach/models/configs/wv_3d/align=0.1.yaml 

python RESN_test.py --model_config=/net/scratch/tianh/explain_teach/models/configs/wv_3d/align=0.0.yaml 

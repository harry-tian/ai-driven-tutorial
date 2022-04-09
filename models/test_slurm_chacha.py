from slurmpy import Slurm

def get_slurm():
    # if segment_len <= 512:
    #     constraint = "rtx2080ti"
    # elif test:
    #     constraint = "rtx2080ti"
    # else:
        # constraint = "rtx8000"
    constraint = "rtx2080ti"
    sbatch_arguments = {"account": "chacha", 
        "partition": "cdac-contrib", # "dev"
        "mem": "50gb",
        "time": "04:00:00",
        "output": "/home/chacha/slurm/out/explain_teach%j.%N.stdout",
        "error": "/home/chacha/slurm/out/explain_teach%j.%N.stderr",
        "chdir": "/home/chacha/slurm",
        "gpus-per-node": f"{1}",
        "ntasks": "1",
        "nodes": "1",
        # "nodelist": "a007",
        "constraint": constraint,
        "signal": "SIGUSR1@120"}

    # if dependent_job:
    #     sbatch_arguments.update({"dependency": f"afterany:{dependent_job}"})

    s = Slurm("explain_teach", sbatch_arguments, bash_strict=False)
    return s

# experiments = [
#     ('mortality', '24', 'all_but_discharge'),
#     ('readmission', 'retro', 'discharge'),
#     ('readmission', 'retro', 'all')
# ]

# segment_mapping = {318:512, 2500:3000}

# balanced_flag = [False, True]
# # structured = True
# structured_flag = [True, ]
# models = [
#     # ("emilyalsentzer/Bio_ClinicalBERT", 318),
#     # ("emilyalsentzer/Bio_Discharge_Summary_BERT", 318),
#     # ("bert-base-uncased", 318),
#     ("roberta-base", 318),
#     ("microsoft/deberta-v3-base", 318),
#     # ("allenai/longformer-base-4096", 2500)
#     ]
# learning_rates = [
#     "5e-3", "1e-3", "5e-4",
#     # "5e-4", "5e-5", "5e-6"
#     ]
#####
# Important
# n_gpu = 4
# train = True
# test = True
#####
# print(f"Training: {train}, Testing: {test}")
# splits = ['train', 'valid', 'test']
# TODO: add mem-per-gpu flag


command = f"""
    source ~/.bashrc
    whoami
    conda activate teaching
    cd /net/scratch/chacha/explain_teach/models
    srun python resn_args.py \
    --embed_dim=10 \
    --wandb_mode=online \
    --wandb_group=resn-emb2 \
    --output_dir=results/resn-emb2 \
    --train_dir=/net/scratch/hanliu-shared/data/bm/train \
    --valid_dir=/net/scratch/hanliu-shared/data/bm/valid \
    --dataloader_num_workers=4 \
    --gpus=1 \
    --seed=42 \
    --max_epochs=100 \
    --learning_rate=1e-4 \
    --vertical_flip=0.5 \
    --rotate=30 \
    --scale=0.2 \
    --train_batch_size=160 \
    --do_train \
    --pretrained"""

    # source /opt/conda/etc/profile.d/conda.sh
    # conda activate clinical-bert

    # cd /home/chaochunh/investigate-clinicalbert

    # srun python -m models.transformers.main \
    # --task {task} \
    # --period {period} \
    # --note_type {note_type} \
    # --num_labels 2 \
    # --max_epochs 10 \
    # --max_seq_length {max_seq_length} \
    # --segment_len {segment_length} \
    # --output_dir /net/scratch/chaochunh/investigate-clinicalbert \
    # --data_dir /net/scratch/chaochunh/test_mimic_output \
    # --model_name_or_path {model} \
    # --warmup_steps 1000 \
    # --cache_dir /net/scratch/chaochunh/transformers \
    # --train_batch_size 8 \
    # --eval_batch_size 8 \
    # --learning_rate {lr} \
    # --fp16"""
s = get_slurm()
s.run(command)
# for task, period, note_type in experiments:
#     for structured in structured_flag:
#         for balanced in balanced_flag:
#             for model, segment_length in models:
#                 for lr in learning_rates:
#                     max_seq_length = segment_mapping[segment_length]
#                     command = f"""
#     source /opt/conda/etc/profile.d/conda.sh
#     conda activate clinical-bert

#     cd /home/chaochunh/investigate-clinicalbert

#     srun python -m models.transformers.main \
#     --task {task} \
#     --period {period} \
#     --note_type {note_type} \
#     --num_labels 2 \
#     --max_epochs 10 \
#     --max_seq_length {max_seq_length} \
#     --segment_len {segment_length} \
#     --output_dir /net/scratch/chaochunh/investigate-clinicalbert \
#     --data_dir /net/scratch/chaochunh/test_mimic_output \
#     --model_name_or_path {model} \
#     --warmup_steps 1000 \
#     --cache_dir /net/scratch/chaochunh/transformers \
#     --train_batch_size 8 \
#     --eval_batch_size 8 \
#     --learning_rate {lr} \
#     --fp16"""
#                     if balanced:
#                         command += " --balanced"
#                     if structured: 
#                         command += " --structured"

#                     if train:
#                         s = get_slurm(segment_length)
#                         command += f" --gpus {n_gpu}"
#                         command += " --do_train"
#                         command += " --overwrite_dir"
#                         job_id = s.run(command)
#                         # submit dependent job after training compelete
#                         if test:
#                             command = command.replace(" --overwrite_dir", "")
#                             command = command.replace(f" --gpus {n_gpu}", " --gpus 1")
#                             command = command.replace(" --do_train", " --do_predict")
#                             command += " --offline"
#                             s_dependent = get_slurm(segment_length, test=True, dependent_job=job_id)
#                             s_dependent.run(command)
#                     elif test:
#                         s = get_slurm(segment_length, test=True)
#                         command += f" --gpus 1"
#                         command += " --offline"
#                         command += " --do_predict"
#                         s.run(command)
#                     else:
                        # raise("Neither train or test is applied.")
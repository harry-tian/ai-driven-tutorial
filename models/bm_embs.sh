#!/bin/bash
runs=(20t3790p 36l4zlp0 3u9xf9rl 2n7yejtu 12sr4qik 11stv4lr)
name=MTL0
for i in ${!runs[@]}; do
    echo $i Getting embeds of $name run: ${runs[$i]}
    python gen_embs.py \
        --model_name=MTL_han \
        --wandb_group=bm_prolific_d50_e100 \
        --wandb_run=${runs[$i]} \
        --wandb_name=$name \
        --suffix=emb50_s$i \
        --dataset=bm \
        --subdir=bm/prolific
done

runs=(u88cd6fd 2mpy7xg5 6m5tzxqu 1dmjba2n qvpxfvlo 186p95gs)
name=MTL1
for i in ${!runs[@]}; do
    echo $i Getting embeds of $name run: ${runs[$i]}
    python gen_embs.py \
        --model_name=MTL_han \
        --wandb_group=bm_prolific_d50_e100 \
        --wandb_run=${runs[$i]} \
        --wandb_name=$name \
        --suffix=emb50_s$i \
        --dataset=bm \
        --subdir=bm/prolific
done

runs=(1ehtqo6u 3ej43usy y5u2qw21 2ool9n2l 1digre3t 1bocgdq1)
name=MTL0.5
for i in ${!runs[@]}; do
    echo $i Getting embeds of $name run: ${runs[$i]}
    python gen_embs.py \
        --model_name=MTL_han \
        --wandb_group=bm_prolific_d50_e100 \
        --wandb_run=${runs[$i]} \
        --wandb_name=$name \
        --suffix=emb50_s$i \
        --dataset=bm \
        --subdir=bm/prolific
done

runs=(3lwngvwq ghixufur 22eagdbt 2yjgnk4z e0br4iu5 lhuzbncb)
name=MTL0.2
for i in ${!runs[@]}; do
    echo $i Getting embeds of $name run: ${runs[$i]}
    python gen_embs.py \
        --model_name=MTL_han \
        --wandb_group=bm_prolific_d50_e100 \
        --wandb_run=${runs[$i]} \
        --wandb_name=$name \
        --suffix=emb50_s$i \
        --dataset=bm \
        --subdir=bm/prolific
done

runs=(1xps8ska 2ikw0ens 2pr0mq9r 2yc0v5vk 34iakqai nusy7ws8)
name=MTL0.8
for i in ${!runs[@]}; do
    echo $i Getting embeds of $name run: ${runs[$i]}
    python gen_embs.py \
        --model_name=MTL_han \
        --wandb_group=bm_prolific_d50_e100 \
        --wandb_run=${runs[$i]} \
        --wandb_name=$name \
        --suffix=emb50_s$i \
        --dataset=bm \
        --subdir=bm/prolific
done
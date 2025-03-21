#!/bin/bash

 FORCE_TORCHRUN=1 
 HOSTFILE="./hostfile" 
 INCLUDE="self_sft_1:0,1,2,3,4,5,6,7@self_sft_2:0,1,2,3,4,5,6,7@self_sft_3:0,1,2,3,4,5,6,7" 
 MASTER_PORT=30252 

 DISTRIBUTED_ARGS="
     --hostfile $HOSTFILE \
     --include $INCLUDE \
     --master_port $MASTER_PORT
 "
#GPUS_PER_NODE=8
#NNODES=1
#NODE_RANK=0
#MASTER_ADDR=localhost
#MASTER_PORT=6001

#DISTRIBUTED_ARGS="
#    --num_gpus $GPUS_PER_NODE \
#    --num_nodes $NNODES \
#    --master_addr $MASTER_ADDR \
#    --master_port $MASTER_PORT
#"

# llamafactory-cli train examples/train_full/qwen2_full_sft_ds2_ii.yaml 
# /mnt/tenant-home_speed/huangshihui/Qwen2.5-7B-Pretrain-Merge
# /mnt/tenant-home_speed/linyun/model/4001_qwen2half_7b/only_telecom_50percent_full_5gnr_gongfu_dt_new_ume/
deepspeed $DISTRIBUTED_ARGS src/train.py \
       --model_name_or_path /mnt/tenant-home_speed/AIM/model/Qwen2.5-14B-Instruct/ \
       --stage sft \
       --do_train True \
       --finetuning_type full \
       --deepspeed examples/deepspeed/ds_z3_config.json \
       --dataset self_sft_g1v21 \
       --template qwen \
       --cutoff_len 16384 \
       --max_samples 100000000 \
       --overwrite_cache True \
       --preprocessing_num_workers 16 \
       --output_dir /mnt/tenant-home_speed/dcr/train/14_qwen_template/ \
       --logging_steps 2 \
       --save_steps 100000 \
       --plot_loss True \
       --overwrite_output_dir True \
       --per_device_train_batch_size 1 \
       --gradient_accumulation_steps 1 \
       --learning_rate 5.0e-6 \
       --num_train_epochs 100.0 \
       --lr_scheduler_type cosine \
       --max_grad_norm 1.0 \
       --warmup_ratio 0.1 \
       --adam_beta2 0.95 \
       --bf16 True \
       --ddp_timeout 180000000 \
       --val_size 0.01 \
       --per_device_eval_batch_size 1 \
       --eval_strategy steps \
       --eval_steps 20 \
       --seed 66 \
       --report_to 'tensorboard'

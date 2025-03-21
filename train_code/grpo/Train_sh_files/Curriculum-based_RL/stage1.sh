cd /mnt/tenant-home_speed/dhl/VLM-R1-main/src/open-r1-multimodal/

export DEBUG_MODE="true"
# export CUDA_VISIBLE_DEVICES=4,5,6,7

RUN_NAME="Qwen2.5-VL-3B-3_tasks_4000"  # /mnt/tenant-home_speed/AIM/model/Qwen2.5-VL-3B-Instruct
export LOG_PATH="/mnt/tenant-home_speed/dhl/VLM-R1-main/Output_Curriculum-based_RL/debug_log_$RUN_NAME.txt"
export WANDB_PROJECT=RUN_NAME


# 创建目录
mkdir -p /mnt/tenant-home_speed/dhl/VLM-R1-main/Output_Curriculum-based_RL/Stage1_True_or_False/Qwen2.5-VL-3B-3_tasks_4000

# 设置环境变量
export WANDB_MODE=offline  # 如果需要离线模式
export WANDB_DIR=/mnt/tenant-home_speed/dhl/VLM-R1-main/Output_Curriculum-based_RL/Stage1_True_or_False/Qwen2.5-VL-3B-3_tasks_4000

# 大概训 600steps就行
torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12347" \
    /mnt/tenant-home_speed/dhl/VLM-R1-main/src/open-r1-multimodal/src/open_r1/Stage1_judge.py \
    --deepspeed /mnt/tenant-home_speed/dhl/VLM-R1-main/src/open-r1-multimodal/local_scripts/zero3.json \
    --output_dir /mnt/tenant-home_speed/dhl/VLM-R1-main/Output_Curriculum-based_RL/Stage1_True_or_False/Qwen2.5-VL-3B-3_tasks_4000 \
    --model_name_or_path /mnt/tenant-home_speed/AIM/model/Qwen2.5-VL-3B-Instruct \
    --dataset_name /mnt/tenant-home_speed/dhl/VLM-R1-main/data_config/stage1_judge/3_tasks_4000.yaml \
    --image_root /mnt/tenant-home_speed/dhl/RL_VL3B/data/ \
    --max_prompt_length 2048 \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --max_pixels 401408 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true
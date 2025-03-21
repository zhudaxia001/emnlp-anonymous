NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MAX_PIXELS=401408 swift sft \
    --model_type qwen2_5_vl \
    --model /mnt/tenant-home_speed/dhl/VLM-R1-main/Output_Curriculum-based_RL_math_resize/Stage3_Open/Stage3_Qwen2.5-VL-3B-GRPO-3_tasks_3000/checkpoint-2300 \
    --dataset /mnt/tenant-home_speed/dhl/VLM-R1-main/Reject_sft/json_files/cat_all_formatted.jsonl \
    --train_type full \
    --torch_dtype bfloat16 \
    --max_steps 2000 \
    --max_length 8192 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-7 \
    --gradient_accumulation_steps 4 \
    --eval_steps 100 \
    --save_steps 100 \
    --freeze_vit False \
    --save_only_model true \
    --save_total_limit 25 \
    --logging_steps 5 \
    --output_dir /mnt/tenant-home_speed/dhl/VLM-R1-main/Reject_sft/small_learn_rate_Output \
    --warmup_ratio 0.05 \
    --system 'You are a helpful assistant' \
    --deepspeed zero3 \


# --attn_impl flash_attn \

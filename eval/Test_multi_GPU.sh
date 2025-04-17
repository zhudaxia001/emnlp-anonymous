
# 正式
python /mnt/tenant-home_speed/dhl/VLM-R1-main/src/eval/muti_process_eval.py --sample_num 1000 --BSZ 84 --test False --task_list detection classify math --model_path /mnt/tenant-home_speed/dhl/VLM-R1-main/Restart_Output/Qwen2.5-VL-3B-GRPO-det_classify_math_v2_iou

# 调试 debug
python /mnt/tenant-home_speed/dhl/VLM-R1-main/src/eval/muti_process_eval.py --sample_num 20 --BSZ 4 --task_list detection classify math --test True --model_path /mnt/tenant-home_speed/dhl/VLM-R1-main/Restart_Output/Qwen2.5-VL-3B-GRPO-det_classify_math_v2_iou

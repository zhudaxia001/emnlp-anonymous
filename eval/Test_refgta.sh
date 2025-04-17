#!/bin/bash

BASE_DIR="/mnt/tenant-home_speed/dhl/VLM-R1-main"
CHECKPOINT_DIR="${BASE_DIR}/Output_Curriculum-based_RL_math_resize/Stage3_Open/Stage3_Qwen2.5-VL-3B-GRPO-3_tasks_3000"
OUTPUT_DIR='/mnt/tenant-home_speed/dhl/VLM-R1-main/Test/sft_data/v1-20250224-102450/detection_out'


# 并行运行所有评估任务 八卡并行测试
python ${BASE_DIR}/src/eval/test_rec_r1.py --BSZ 32 --checkpoint_dir ${CHECKPOINT_DIR} --steps 100 --gpu 0 --sample_num 1000 &
python ${BASE_DIR}/src/eval/test_rec_r1.py --BSZ 32 --checkpoint_dir ${CHECKPOINT_DIR} --steps 200 --gpu 1 --sample_num 1000 &
python ${BASE_DIR}/src/eval/test_rec_r1.py --BSZ 32 --checkpoint_dir ${CHECKPOINT_DIR} --steps 300 --gpu 2 --sample_num 1000 &
python ${BASE_DIR}/src/eval/test_rec_r1.py --BSZ 32 --checkpoint_dir ${CHECKPOINT_DIR} --steps 400 --gpu 3 --sample_num 1000 &
python ${BASE_DIR}/src/eval/test_rec_r1.py --BSZ 32 --checkpoint_dir ${CHECKPOINT_DIR} --steps 500 --gpu 4 --sample_num 1000 &
python ${BASE_DIR}/src/eval/test_rec_r1.py --BSZ 32 --checkpoint_dir ${CHECKPOINT_DIR} --steps 600 --gpu 5 --sample_num 1000 &
python ${BASE_DIR}/src/eval/test_rec_r1.py --BSZ 32 --checkpoint_dir ${CHECKPOINT_DIR} --steps 700 --gpu 6 --sample_num 1000 &
python ${BASE_DIR}/src/eval/test_rec_r1.py --BSZ 32 --checkpoint_dir ${CHECKPOINT_DIR} --steps 1700 --gpu 7 --sample_num 1000 &

# 等待所有后台任务完成
wait

echo "所有评估任务已完成!"

#!/bin/bash

declare -A TASKS=(
    # 只保留Stage2的配置，因为我们只需要监控Stage2
    ["stage2"]="/mnt/tenant-home_speed/dhl/VLM-R1-main/Output_Curriculum-based_RL_math_resize/Stage2_Multi_choice/Stage2_Qwen2.5-VL-3B-GRPO-3_tasks_3000:12348:Stage2_choice_math_resize.py:/mnt/tenant-home_speed/dhl/VLM-R1-main/Train_sh_files/Curriculum-based_RL_math_resize/stage3_math_resize.sh"
)

TARGET_STEP=600

monitor_and_switch() {
    local output_dir=$1
    local port=$2
    local script_name=$3
    local next_script=$4
    
    echo "Monitoring $output_dir"
    echo "Current task: $script_name (Port: $port)"
    echo "Next task will be: $next_script"
    
    while true; do
        # 检查checkpoint
        if [ -d "${output_dir}/checkpoint-${TARGET_STEP}" ]; then
            echo "Found checkpoint-${TARGET_STEP} in ${output_dir}!"
            
            echo "Terminating all python processes..."
            # 使用更可靠的方式杀掉所有Python进程
            ps aux | grep python | grep -v grep | sort -k 3 -r | head -n 8 | awk '{print $2}' | xargs kill -9
            
            # 等待进程完全终止
            sleep 20
            
            # 确认进程已终止
            if ps aux | grep "torchrun" | grep -v grep > /dev/null; then
                echo "Warning: Some processes are still running, trying again..."
                ps aux | grep python | grep -v grep | sort -k 3 -r | head -n 8 | awk '{print $2}' | xargs kill -9
                sleep 10
            fi
            
            # 启动下一个训练脚本
            echo "Starting next training task: $next_script"
            sh $next_script &
            
            return 0
        fi
        
        current_time=$(date "+%Y-%m-%d %H:%M:%S")
        echo -n "Checking... ($current_time)\r"
        sleep 30
    done
}

# 开始监控Stage2
echo "Starting Stage2 monitoring..."
IFS=':' read -r output_dir port script_name next_script <<< "${TASKS[stage2]}"
monitor_and_switch "$output_dir" "$port" "$script_name" "$next_script"

echo "All tasks completed!"
#!/bin/bash

# 训练任务配置
# declare -A TASKS=(
#     ["task1"]="第一阶段输出目录1:训练脚本端口号1:下一个训练脚本 stage2"
#     ["task2"]="第二阶段输出目录:训练脚本端口号2:下一个训练脚本 stage3"
# )



# declare -A TASKS=(
#     ["task1"]="/mnt/tenant-home_speed/dhl/VLM-R1-main/Output_Curriculum-based_RL/Stage1_True_or_False/Qwen2.5-VL-3B-3_tasks_4000:12347:/mnt/tenant-home_speed/dhl/VLM-R1-main/Train_sh_files/Curriculum-based_RL/stage2.sh"
#     ["task2"]="/mnt/tenant-home_speed/dhl/VLM-R1-main/Output_Curriculum-based_RL/Stage2_Multi_choice/Stage2_Qwen2.5-VL-3B-GRPO-3_tasks_3000:12348:/mnt/tenant-home_speed/dhl/VLM-R1-main/Train_sh_files/Curriculum-based_RL/stage3.sh"
# )

# TARGET_STEP=600

# monitor_and_switch() {
#     local output_dir=$1
#     local port=$2
#     local next_script=$3
    
#     echo "Monitoring $output_dir (Port: $port)"
    
#     while true; do
#         # 检查checkpoint
#         if [ -d "${output_dir}/checkpoint-${TARGET_STEP}" ]; then
#             echo "Found checkpoint-${TARGET_STEP} in ${output_dir}!"
            
#             # 查找并显示进程信息
#             pid=$(ps aux | \
#                   grep "torchrun" | \
#                   grep "Stage1_judge.py" | \
#                   grep "master_port=\"*${port}\"*" | \
#                   grep -v "grep" | \
#                   awk '{print $2}')
            
#             if [ ! -z "$pid" ]; then
#                 echo "Found process PID: $pid"
#                 echo "Process details:"
#                 ps -fp $pid
                
#                 echo "Terminating process..."
#                 # pkill -TERM -P $pid
#                 # kill -TERM -$pid 2>/dev/null
                
#                 # 等待进程完全终止
#                 sleep 10
                
#                 # 启动下一个训练脚本
#                 echo "Starting next training task: $next_script"
#                 # sh $next_script &
                
#                 # 返回新任务的PID
#                 echo $!
#                 break
#             fi
#         fi
        
#         current_time=$(date "+%Y-%m-%d %H:%M:%S")
#         echo -n "Checking... ($current_time)\r"
#         sleep 30
#     done
# }

# # 并行监控多个任务
# for task_name in "${!TASKS[@]}"; do
#     IFS=':' read -r output_dir port next_script <<< "${TASKS[$task_name]}"
    
#     # 在后台启动监控
#     monitor_and_switch "$output_dir" "$port" "$next_script" &
    
#     # 存储监控进程的PID
#     monitor_pids+=($!)
    
#     echo "Started monitoring for $task_name (PID: ${monitor_pids[-1]})"
# done

# # 等待所有监控进程完成
# wait


#!/bin/bash

# 任务定义，每个任务包含4个参数，用冒号分隔：
# 1. output_dir: 输出目录，用于检查checkpoint
# 2. port: 当前任务使用的端口号
# 3. script_name: 当前运行的脚本名称（用于查找进程）
# 4. next_script: 下一个要执行的脚本路径
declare -A TASKS=(
    # Stage1任务配置
    ["stage1"]="/mnt/tenant-home_speed/dhl/VLM-R1-main/Output_Curriculum-based_RL_math_resize/Stage1_True_or_False/Qwen2.5-VL-3B-3_tasks_4000:12347:Stage1_judge_math_resize.py:/mnt/tenant-home_speed/dhl/VLM-R1-main/Train_sh_files/Curriculum-based_RL_math_resize/stage2_math_resize.sh"
    
    # Stage2任务配置
    ["stage2"]="/mnt/tenant-home_speed/dhl/VLM-R1-main/Output_Curriculum-based_RL_math_resize/Stage2_Multi_choice/Stage2_Qwen2.5-VL-3B-GRPO-3_tasks_3000:12348:Stage2_choice_math_resize.py:/mnt/tenant-home_speed/dhl/VLM-R1-main/Train_sh_files/Curriculum-based_RL_math_resize/stage3_math_resize.sh"
)


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
            
            # 查找当前任务的进程
            pid=$(ps aux | \
                  grep "torchrun" | \
                  grep "$script_name" | \
                  grep "master_port=\"*${port}\"*" | \
                  grep -v "grep" | \
                  awk '{print $2}')
            
            if [ ! -z "$pid" ]; then
                echo "Found process PID: $pid"
                echo "Process details:"
                ps -fp $pid
                
                echo "Terminating process..."
                top -b -n 1 | grep python | head -n 8 | awk '{print $1}' | xargs kill -9

                # 等待进程完全终止
                sleep 20
                
                # 启动下一个训练脚本
                echo "Starting next training task: $next_script"
                sh $next_script &
                
                return 0  # 成功完成当前任务
            fi
        fi
        
        current_time=$(date "+%Y-%m-%d %H:%M:%S")
        echo -n "Checking... ($current_time)\r"
        sleep 30
    done
}

# 按顺序执行任务
echo "Starting sequential task monitoring..."

# 先监控Stage1
IFS=':' read -r output_dir port script_name next_script <<< "${TASKS[stage1]}"
monitor_and_switch "$output_dir" "$port" "$script_name" "$next_script"

# # 等待一段时间确保Stage2开始运行
sleep 30

# 再监控Stage2
IFS=':' read -r output_dir port script_name next_script <<< "${TASKS[stage2]}"
monitor_and_switch "$output_dir" "$port" "$script_name" "$next_script"

echo "All tasks completed!"
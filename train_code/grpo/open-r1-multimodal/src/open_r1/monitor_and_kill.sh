#!/bin/bash

OUTPUT_DIR="/mnt/tenant-home_speed/dhl/VLM-R1-main/Output_Curriculum-based_RL/Stage1_True_or_False/Qwen2.5-VL-3B-3_tasks_4000"
TARGET_STEP=600
PORT="$1"

echo "Starting monitoring for checkpoint-${TARGET_STEP} in ${OUTPUT_DIR}"
echo "Looking for process with port ${PORT}"

find_and_kill_process() {
    # 查找进程PID
    pid=$(ps aux | \
          grep "torchrun" | \
          grep "Stage1_judge.py" | \
          grep "master_port=\"*${PORT}\"*" | \
          grep -v "grep" | \
          awk '{print $2}')
    
    if [ ! -z "$pid" ]; then
        echo "Found process PID: $pid"
        echo "Process details:"
        ps -fp $pid
        
        echo "Command line:"
        cat /proc/$pid/cmdline | tr '\0' ' '
        echo -e "\n"
        
        echo "Start time:"
        ls -ld /proc/$pid
        
        # 杀进程
        # echo "Killing process group $pid..."
        # pkill -TERM -P $pid
        # kill -TERM -$pid 2>/dev/null
        # echo "Process terminated"
        echo "Process terminated"
    else
        echo "No matching process found"
    fi
}

while true; do
    if [ -d "${OUTPUT_DIR}/checkpoint-${TARGET_STEP}" ]; then
        echo "Found checkpoint-${TARGET_STEP}!"
        find_and_kill_process
        break
    fi
    
    current_time=$(date "+%Y-%m-%d %H:%M:%S")
    echo -n "Checking... ($current_time)\r"
    sleep 30
done
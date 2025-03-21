#!/bin/bash

PORT="$1"

# 只查找并输出PID
pid=$(ps aux | \
      grep "torchrun" | \
      grep "Stage1_judge.py" | \
      grep "master_port=\"*${PORT}\"*" | \
      grep -v "grep" | \
      awk '{print $2}')

if [ ! -z "$pid" ]; then
    echo "$pid"
else
    echo "No matching process found"
fi
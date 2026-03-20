#!/bin/bash
# 监控当前 v7 实验完成后，用修复后的代码重新跑实验
# 用法: nohup bash run_monitor.sh &

LOG="/root/FL-optimize/monitor_run.log"
RESULT_JSON="/root/FL-optimize/results/UNSW-NB15_fedpcnn_non-iid_multi_alpha0.5.json"

echo "[$(date)] 监控启动，等待当前实验完成..." | tee -a $LOG

# 等待当前实验完成（nova-mist 进程退出）
while pgrep -f "python3 main.py --model fedpcnn --dataset UNSW-NB15" > /dev/null 2>&1; do
    sleep 30
done

echo "[$(date)] 当前实验已完成" | tee -a $LOG

# 记录旧结果
if [ -f "$RESULT_JSON" ]; then
    echo "[$(date)] 旧结果:" | tee -a $LOG
    cat $RESULT_JSON | tee -a $LOG
fi

echo "[$(date)] 开始用修复后的代码跑新实验..." | tee -a $LOG

# 跑新实验
cd /root/FL-optimize
python3 main.py \
    --model fedpcnn \
    --dataset UNSW-NB15 \
    --classification multi \
    --partition non-iid \
    --alpha 0.5 \
    --seed 42 \
    2>&1 | tee -a $LOG

echo "[$(date)] 新实验完成" | tee -a $LOG

# 记录新结果
if [ -f "$RESULT_JSON" ]; then
    echo "[$(date)] 新结果:" | tee -a $LOG
    cat $RESULT_JSON | tee -a $LOG
fi

echo "[$(date)] 全部完成" | tee -a $LOG

#!/bin/bash

# 设置默认参数
NUM_TASKS=50
CLASS_PER_TASK=20
BATCH_SIZE=2
NUM_EPOCHS_PER_TASK=30
LR=0.00005
NUM_WORKERS=8
SAVE_PATH="/home/biye/code/02-NCP-1/results/Grad-vit-checkpoints-last_iter/"
DATASET_ROOT="/home/biye/dataset/ILSVRC2012_split"
ARCH="vit-base"

# 启动训练
python /home/work/user-job-dir/app/02-NCP-1/LG-Extract/ViT-Grad.py \
    --num-tasks $NUM_TASKS \
    --class-per-task $CLASS_PER_TASK \
    --batch-size $BATCH_SIZE \
    --num-epochs-per-task $NUM_EPOCHS_PER_TASK \
    --lr $LR \
    --num-workers $NUM_WORKERS \
    --save-path $SAVE_PATH \
    --dataset-root $DATASET_ROOT \
    --arch $ARCH

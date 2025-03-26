#!/bin/bash

# 设置默认参数
NUM_TASKS=50
CLASS_PER_TASK=20
BATCH_SIZE=32
NUM_EPOCHS_PER_TASK=30
LR=0.0001
NUM_WORKERS=4
SAVE_PATH="./results/Grad-cnn-checkpoints/"
DATASET_ROOT="./imagenet"
ARCH="resnet34"

# 启动训练
python train.py \
    --num-tasks $NUM_TASKS \
    --class-per-task $CLASS_PER_TASK \
    --batch-size $BATCH_SIZE \
    --num-epochs-per-task $NUM_EPOCHS_PER_TASK \
    --lr $LR \
    --num-workers $NUM_WORKERS \
    --save-path $SAVE_PATH \
    --dataset-root $DATASET_ROOT \
    --arch $ARCH

#!/bin/bash

# List of folders
evalset=(
    # movi_a_0004_anoMask
    movi_a_batchGS_01_50_fcnet
)

DATA_DIR=/home/skyworker/result/4DGS_SlotAttention/shape_of_motion
OUT_DIR=/home/skyworker/result/4DGS_SlotAttention/slot_4dgs

# Batch run preprocess
for seq in ${evalset[@]}; do
    python train.py \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR/$seq \
    --batch_size 72 \
    --num_slots 10 \
    --num_iterations 3 \
    --num_workers 0 \
    --num_epochs 1000
done
#!/bin/bash

# List of folders
evalset=(
    movi_a_0001_anoMask
)

DATA_DIR=/home/skyworker/result/4DGS_SlotAttention/shape_of_motion
OUT_DIR=/home/skyworker/result/4DGS_SlotAttention/shape_of_motion
# Batch run preprocess
for seq in ${evalset[@]}; do
    python kmeans_test.py \
    --data_dir $DATA_DIR/$seq \
    --output_dir $OUT_DIR/${seq}_slotTest \
    --num_slots 8 \
    --num_iterations 10000\
    --frame 2
done


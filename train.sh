#!/bin/bash

# List of folders
evalset=(
    # movi_a_0003_anoMask
    # movi_a_single
    # movi_a_mix
    clevr_gpu
)

# DATA_DIR=/home/skyworker/result/4DGS_SlotAttention/shape_of_motion
DATA_DIR=/home/skyworker/temp
OUT_DIR=/home/skyworker/result/4DGS_SlotAttention/slot_4dgs
CLEVR_DIR=/home/skyworker/tensorflow_datasets/downloads/extracted/ZIP.dl.fbaipublicfiles.com_clevr_CLEVR_v1.0XNYc8Qlu0glE35PJrbMedNGJuEWalPVLoACQ5cWZNtE.zip/CLEVR_v1.0/images/train

# Batch run preprocess
for seq in ${evalset[@]}; do
    python train.py \
    --data_dir $CLEVR_DIR \
    --output_dir $OUT_DIR/$seq \
    --batch_size 64 \
    --num_slots 10 \
    --num_iterations 3 \
    --num_workers 0 \
    --num_epochs 100
done

    --data_dir $CLEVR_DIR \
    --data_dir $DATA_DIR/$seq/images \
    
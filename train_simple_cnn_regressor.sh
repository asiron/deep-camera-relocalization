#!/bin/bash

export CUDA_PATH=/usr/local/cuda-8.0/bin
export PATH=${CUDA_PATH}${PATH:+:${PATH}}

export LOCAL_CUDNN_PATH=~/cudnn/
export LD_LIBRARY_PATH=$LOCAL_CUDNN_PATH/cuda/lib64:$LD_LIBRARY_PATH

DATASET_DIR=.

FEATURES=(
  "${DATASET_DIR}/extracted-inception-v3-position00/features.npy"
  "${DATASET_DIR}/extracted-inception-v3-position01/features.npy"
)

LABELS=(
  "../../datasets/wing/position00/seq00/labels"
  "../../datasets/wing/position01/seq00/labels"
)

LOSS=naive-w
OUTPUT_DIR="new-test-runs/${LOSS}/-`date +"%m_%d_%Y--%H-%M-%S"`"

python train_simple_cnn_regressor.py \
  -l "${LABELS[@]}" \
  -f "${FEATURES[@]}" \
  -o "${OUTPUT_DIR}" \
  -i 200 \
  -m "${LOSS}"

# python train_simple_cnn_regressor.py \
#   -l "${LABELS[@]}" \
#   -f "${FEATURES[@]}" \
#   -o "${OUTPUT_DIR}" \
#   -i 200 \
#   -m naive-p \
#   -c test-run-proper-09_30_2017--04-14-01/checkpoints/L1,beta=117.2,lr=5.08e-05,dropout=0.19,l2_regu=0.13/weights.0999-0.5054.hdf5


#!/bin/bash

export CUDA_PATH=/usr/local/cuda-8.0/bin
export PATH=${CUDA_PATH}${PATH:+:${PATH}}

export LOCAL_CUDNN_PATH=~/cudnn/
export LD_LIBRARY_PATH=$LOCAL_CUDNN_PATH/cuda/lib64:$LD_LIBRARY_PATH

DATASET_DIR="../../datasets/wing"

PRETRAINED_NET="googlenet"
PRETRAINED_DATASET="places365"

FEATURES=(
  "${DATASET_DIR}/position00/seq00/extracted_features/${PRETRAINED_NET}/${PRETRAINED_DATASET}/cnn_features.npy"
  "${DATASET_DIR}/position01/seq00/extracted_features/${PRETRAINED_NET}/${PRETRAINED_DATASET}/cnn_features.npy"
)

LABELS=(
  "${DATASET_DIR}/position00/seq00/labels"
  "${DATASET_DIR}/position01/seq00/labels"
)

HYPERPARAM_CONFIG="configs.hyperparam_initial_config"

MODE=initial

TOP_MODEL_TYPE=regressor
LOSS=naive_weighted

DATE=`date +"%m_%d_%Y--%H-%M-%S"`
OUTPUT_DIR="${HOME}/pose-regression/experiments"
OUTPUT_DIR="${OUTPUT_DIR}/${MODE}-${TOP_MODEL_TYPE}-${LOSS}-${PRETRAINED_NET}-${PRETRAINED_DATASET}/${DATE}"

python train.py \
  -l "${LABELS[@]}" \
  -f "${FEATURES[@]}" \
  -o "${OUTPUT_DIR}" \
  --mode "${MODE}" \
  --top-model-type "${TOP_MODEL_TYPE}" \
  --loss "${LOSS}" \
  --hyperparam-config "${HYPERPARAM_CONFIG}" \
  -i 1

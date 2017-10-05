#!/bin/bash

export CUDA_PATH=/usr/local/cuda-8.0/bin
export PATH=${CUDA_PATH}${PATH:+:${PATH}}
export LOCAL_CUDNN_PATH=~/cudnn/
export LD_LIBRARY_PATH=$LOCAL_CUDNN_PATH/cuda/lib64:$LD_LIBRARY_PATH

DATE=`date +"%m_%d_%Y--%H-%M-%S"`

DATASET_DIR="../../datasets/wing"

LABELS=(
  "${DATASET_DIR}/position00/seq00/labels"
  "${DATASET_DIR}/position01/seq00/labels"
)

HYPERPARAM_CONFIG="configs.hyperparam_initial_config"

ITERS=30
MODE=initial
TOP_MODEL_TYPE=regressor

NETS=(
  googlenet,imagenet
  googlenet,places365
  inception_resnet_v2,imagenet
)

for net in "${NETS[@]}"; do 
  IFS=',' read arch dataset <<< "${net}"
  features=(
    "${DATASET_DIR}/position00/seq00/extracted_features/${arch}/${dataset}/cnn_features.npy"
    "${DATASET_DIR}/position01/seq00/extracted_features/${arch}/${dataset}/cnn_features.npy"
  )
  for loss in naive_weighted quaternion_weighted; do 
    OUTPUT_DIR="/media/labuser/Seagate Expansion Drive/experiments/results"
    OUTPUT_DIR="${OUTPUT_DIR}/${MODE}-${TOP_MODEL_TYPE}-${loss}-${arch}-${dataset}/${DATE}"
    for i in `seq $ITERS`; do
      python train.py \
        -l "${LABELS[@]}" \
        -f "${features[@]}" \
        -o "${OUTPUT_DIR}" \
        --mode "${MODE}" \
        --top-model-type "${TOP_MODEL_TYPE}" \
        --loss "${loss}" \
        --hyperparam-config "${HYPERPARAM_CONFIG}" \
        -i 1
    done
  done
done

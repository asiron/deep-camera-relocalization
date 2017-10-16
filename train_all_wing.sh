#!/bin/bash

export CUDA_PATH=/usr/local/cuda-8.0/bin
export PATH=${CUDA_PATH}${PATH:+:${PATH}}
export LOCAL_CUDNN_PATH=~/cudnn/
export LD_LIBRARY_PATH=$LOCAL_CUDNN_PATH/cuda/lib64:$LD_LIBRARY_PATH

DATE=`date +"%m_%d_%Y--%H-%M-%S"`

DATASET_DIR="${HOME}/datasets/wing"

#HYPERPARAM_CONFIG="configs.hyperparam_initial_config"
HYPERPARAM_CONFIG="configs.hyperparam_finetune"

ITERS=20
TOP_MODEL_TYPE=lstm

NETS=(
  #googlenet,imagenet
  googlenet,places365
  #inception_resnet_v2,imagenet
)

MODE=finetune

TOP_MODEL_WEIGHTS=something

LOSSES=(
  # naive_weighted
  # quaternion_error_weighted
  # quaternion_angle_weighted
  quaternion_error_homoscedastic
  quaternion_angle_homoscedastic
  naive_homoscedastic
)

for net in "${NETS[@]}"; do 
  IFS=',' read arch dataset <<< "${net}"
        
  train_seq_label_dirs=()
  valid_seq_label_dirs=() 

  train_seq_feature_dirs=()
  valid_seq_feature_dirs=() 
  
  for seq in ${DATASET_DIR}/position_*/train/seq_*; do
    train_seq_label_dirs+=("${seq}/labels")
    train_seq_feature_dirs+=("${seq}/extracted_features/${arch}/${dataset}/finetune_features.npy")
  done

  for seq in ${DATASET_DIR}/position_*/test/seq_*; do
    valid_seq_label_dirs+=("${seq}/labels")
    valid_seq_feature_dirs+=("${seq}/extracted_features/${arch}/${dataset}/finetune_features.npy")
  done  
  
  for loss in "${LOSSES[@]}"; do     
    OUTPUT_DIR="/media/labuser/Flight_data/maciej-cnn-wing-results-finetune"
    OUTPUT_DIR="${OUTPUT_DIR}/${DATE}/${MODE}-${TOP_MODEL_TYPE}-${loss}-${arch}-${dataset}/"

    mkdir -p "${OUTPUT_DIR}"

    for i in `seq $ITERS`; do
      python train.py \
        -tl "${train_seq_label_dirs[@]}" \
        -tf "${train_seq_feature_dirs[@]}" \
        -vl "${valid_seq_label_dirs[@]}" \
        -vf "${valid_seq_feature_dirs[@]}" \
        -o "${OUTPUT_DIR}" \
        --mode "${MODE}" \
        --top-model-type "${TOP_MODEL_TYPE}" \
        --loss "${loss}" \
        --hyperparam-config "${HYPERPARAM_CONFIG}" \
        --top-model-weights "${TOP_MODEL_WEIGHTS}" \
        --finetuning-model-arch "${arch}" \
        --finetuning-model-dataset "${dataset}" \
        -i 1 \
        --epochs 50 \
        --batch-size 64 \
        --save-period 1 \
        --seq-len 10
      exit
   done
  done
done

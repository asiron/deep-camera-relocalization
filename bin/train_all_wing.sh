#!/bin/bash

# export CUDA_PATH=/usr/local/cuda-8.0/bin
# export PATH=${CUDA_PATH}${PATH:+:${PATH}}
# export LOCAL_CUDNN_PATH=~/cudnn/
# export LD_LIBRARY_PATH=$LOCAL_CUDNN_PATH/cuda/lib64:$LD_LIBRARY_PATH

MODULE="pose_regression.train"

DATE=`date +"%m_%d_%Y--%H-%M-%S"`

DATASET_DIR="${HOME}/datasets/wing"
RESULT_DIR="/media/labuser/Flight_data/maciej-cnn-wing-results-finetune"

MODE=finetune

ITERS=20
TOP_MODEL_TYPE=regressor
HYPERPARAM_CONFIG="pose_regression.configs.hyperparam_finetune"

NETS=(
  #googlenet,imagenet
  googlenet,places365
  #inception_resnet_v2,imagenet
)

TOP_MODEL_WEIGHTS=something

LOSSES=(
  # naive_weighted
  # quaternion_error_weighted
  # quaternion_angle_weighted
  quaternion_angle_homoscedastic
  quaternion_error_homoscedastic
  #naive_homoscedastic
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
  
  for i in `seq $ITERS`; do
    for loss in "${LOSSES[@]}"; do     

      output_dir="${RESULT_DIR}/${DATE}/${MODE}-${TOP_MODEL_TYPE}-${loss}-${arch}-${dataset}/"

      mkdir -p "${output_dir}"

      python -m "${MODULE}" \
        -tl "${train_seq_label_dirs[@]}" \
        -tf "${train_seq_feature_dirs[@]}" \
        -vl "${valid_seq_label_dirs[@]}" \
        -vf "${valid_seq_feature_dirs[@]}" \
        -o "${output_dir}" \
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
        --save-period 1
        #--seq-len 72
    done
  done
done

#!/bin/bash

export CUDA_PATH=/usr/local/cuda-8.0/bin
export PATH=${CUDA_PATH}${PATH:+:${PATH}}
export LOCAL_CUDNN_PATH=~/cudnn/
export LD_LIBRARY_PATH=$LOCAL_CUDNN_PATH/cuda/lib64:$LD_LIBRARY_PATH

DATE=`date +"%m_%d_%Y--%H-%M-%S"`

DATASET_DIR="${HOME}/datasets/7scenes"

SCENES=(
  office
  #redkitchen
  #stairs
)

HYPERPARAM_CONFIG="configs.hyperparam_initial_config"

ITERS=4
TOP_MODEL_TYPE=regressor

NETS=(
  inception_resnet_v2,imagenet
  #googlenet,imagenet
  #googlenet,places365
)

LOSSES=(
  # naive_weighted
  # quaternion_error_weighted
  # quaternion_angle_weighted
  naive_homoscedastic
  quaternion_error_homoscedastic
  quaternion_angle_homoscedastic
)

MODE=initial


for net in "${NETS[@]}"; do 
  IFS=',' read arch dataset <<< "${net}"
    
  for scene in "${SCENES[@]}"; do
    
    scene_dir="${DATASET_DIR}/${scene}"
    echo 'Processing scene:' "${scene_dir}"

    train_seqs=$(cat "${scene_dir}/TrainSplit.txt" | tr '\n\r' ' ')
    #valid_seqs=$(cat "${scene_dir}/ValidSplit.txt" | tr '\n\r' ' ')
    valid_seqs=$(cat "${scene_dir}/TestSplit.txt" | tr '\n\r' ' ')

    train_seq_label_dirs=()
    valid_seq_label_dirs=() 

    train_seq_feature_dirs=()
    valid_seq_feature_dirs=() 
    
    for seq in $train_seqs; do
      train_seq_label_dirs+=(${scene_dir}/${seq}/)
      train_seq_feature_dirs+=(${scene_dir}/${seq}/extracted_features/${arch}/${dataset}/cnn_features.npy)
    done

    for seq in $valid_seqs; do
      valid_seq_label_dirs+=(${scene_dir}/${seq}/)
      valid_seq_feature_dirs+=(${scene_dir}/${seq}/extracted_features/${arch}/${dataset}/cnn_features.npy)
    done

    for loss in "${LOSSES[@]}"; do     
      OUTPUT_DIR="/media/labuser/Flight_data/maciej-cnn-pose-regression-results-initial"
      OUTPUT_DIR="${OUTPUT_DIR}/${DATE}/${scene}/${MODE}-${TOP_MODEL_TYPE}-${loss}-${arch}-${dataset}/"

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
          --save-period 3 \
          --batch-size 128 \
          --epochs 300 \
          -i 1
      done
    done
  done
done

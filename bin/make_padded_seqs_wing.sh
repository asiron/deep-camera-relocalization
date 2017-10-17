#!/bin/bash

MODULE="pose_regression.scripts.make_padded_sequences"

DATASET_DIR="${HOME}/datasets/wing"

NETS=(
  googlenet,imagenet
  googlenet,places365
  inception_resnet_v2,imagenet
)

for net in "${NETS[@]}"; do 
  IFS=',' read arch dataset <<< "${net}"
  
  echo "Processing CNN features from ${arch}-${dataset}"    

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
  
  output_dir="${DATASET_DIR}/extracted_sequences/${arch}/${dataset}/"
  mkdir -p "${output_dir}"

  python -m "${MODULE}" \
    -tl "${train_seq_label_dirs[@]}" \
    -tf "${train_seq_feature_dirs[@]}" \
    -vl "${valid_seq_label_dirs[@]}" \
    -vf "${valid_seq_feature_dirs[@]}" \
    -o "${output_dir}" \
    --seq-len 72

done

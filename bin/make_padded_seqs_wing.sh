#!/bin/bash

MODULE="pose_regression.scripts.make_padded_sequences"


#WING_DATASET=wing
WING_DATASET=wing-5

DATASETS="/media/labuser/Storage/arg-00/datasets"
DATASET_DIR="${DATASETS}/${WING_DATASET}"

FEATURE_TYPES=(
  cnn
  finetune
)

CONFIGS=(
  ##stateful,83,36,-1
  ##standard,83,-1,5
  stateful,72,30,-1
  #standard,83,-1,5
)

NETS=(
  googlenet,imagenet
  googlenet,places365
  inception_resnet_v2,imagenet
)

OUTPUT_DIR="${DATASET_DIR}"

for feature_type in "${FEATURE_TYPES[@]}"; do

  for config in "${CONFIGS[@]}"; do
    IFS=',' read type seq_len batch_size subseq_len <<< "${config}"
    
    echo "Processing config: ${type} ${seq_len} ${batch_size} ${subseq_len}"    

    for net in "${NETS[@]}"; do 
      IFS=',' read arch dataset <<< "${net}"
      
      echo "Processing CNN features from ${arch}-${dataset}"    

      train_seq_label_dirs=()
      valid_seq_label_dirs=() 

      train_seq_feature_dirs=()
      valid_seq_feature_dirs=()
      
      for seq in ${DATASET_DIR}/position_*/train/seq_*; do
        train_seq_label_dirs+=("${seq}/labels")
        train_seq_feature_dirs+=("${seq}/extracted_features/${arch}/${dataset}/${feature_type}_features.npy")
      done

      for seq in ${DATASET_DIR}/position_*/test/seq_*; do
        valid_seq_label_dirs+=("${seq}/labels")
        valid_seq_feature_dirs+=("${seq}/extracted_features/${arch}/${dataset}/${feature_type}_features.npy")
      done  
      
      output_dir="${OUTPUT_DIR}/extracted_sequences/${arch}/${dataset}/"
      mkdir -p "${output_dir}"

      python -m "${MODULE}" \
        -tl "${train_seq_label_dirs[@]}" \
        -tf "${train_seq_feature_dirs[@]}" \
        -vl "${valid_seq_label_dirs[@]}" \
        -vf "${valid_seq_feature_dirs[@]}" \
        -o "${output_dir}" \
        --feature-type "${feature_type}" \
        --seq-len "${seq_len}" \
        --type "${type}" \
        --batch-size "${batch_size}" \
        --subseq-len "${subseq_len}"
    done
  done
done

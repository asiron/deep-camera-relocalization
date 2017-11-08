#!/bin/bash

#TODO CLEAN SEQ_LEN, SUBSEQ_LEN DISAMBIGUATION MESS

MODULE="pose_regression.scripts.make_padded_sequences"

DATASETS="/media/labuser/Storage/arg-00/datasets"
DATASET_DIR="${DATASETS}/7scenes"

SCENES=(
  office
  #redkitchen
)

FEATURE_TYPES=(
  cnn
  #finetune
)
 
CONFIGS=(
  #stateful,100,20,-1
  standard,-1,-1,10,1
)

NETS=(
  #googlenet,imagenet
  #googlenet,places365
  #inception_resnet_v2,imagenet
  vgg16,hybrid1365
)

for scene in "${SCENES[@]}"; do

  for feature_type in "${FEATURE_TYPES[@]}"; do

    for config in "${CONFIGS[@]}"; do
      IFS=',' read type seq_len batch_size subseq_len step <<< "${config}"
      
      echo "Processing config: ${type} ${seq_len} ${batch_size} ${subseq_len}" "${step}"    

      for net in "${NETS[@]}"; do 
        IFS=',' read arch dataset <<< "${net}"
        
        echo "Processing CNN features from ${arch}-${dataset}"

        train_seq_label_dirs=()
        valid_seq_label_dirs=() 

        train_seq_feature_dirs=()
        valid_seq_feature_dirs=()
        
        scene_dir="${DATASET_DIR}/${scene}"

        for seq in ${scene_dir}/train/seq-*/; do
          train_seq_label_dirs+=("${seq}")
          train_seq_feature_dirs+=("${seq}/extracted_features/${arch}/${dataset}/${feature_type}_features.npy")
        done

        for seq in ${scene_dir}/test/seq-*; do
          valid_seq_label_dirs+=("${seq}")
          valid_seq_feature_dirs+=("${seq}/extracted_features/${arch}/${dataset}/${feature_type}_features.npy")
        done  
        
        if [[ "${type}" == "standard" ]]; then
          length="${subseq_len}"
        elif [[ "${type}" == "stateful" ]]; then
          length="${seq_len}"
        fi

        output_dir="${scene_dir}/extracted_sequences/${arch}/${dataset}/${type}/${length}"
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
          --subseq-len "${subseq_len}" \
          --step "${step}"
      done
    done
  done
done
#!/bin/bash

MODULE="pose_regression.train"

DATE=`date +"%m_%d_%Y--%H-%M-%S"`

MODE=finetune
#MODE=initial

DATASETS="/media/labuser/Storage/arg-00/datasets"

DATASET_DIR="${DATASETS}/7scenes"
OUTPUT_DIR="/media/labuser/Flight_data1/maciej-cnn-7scenes-results-${MODE}"

SEQ_LEN=4

ITERS=100
EPOCHS=500

SCENES=(
  # chess
  # fire
  # heads
  # pumpkin
  office
  # stairs
  # redkitchen
)

HYPERPARAMS_MODULES="pose_regression.configs"

TOP_MODEL_TYPES=(
  #stateful-lstm
  #standard-lstm
  spatial-lstm
  #regressor
)

CONFIGS=(
  #googlenet,imagenet,64,settings.adam-no-modifier
  #googlenet,places365,64,settings.adam-no-modifier
  vgg16,hybrid1365,32,settings.adam-no-modifier
  #inception_resnet_v2,imagenet,20,settings.adam-lr-reducer
)

MODEL_WEIGHTS=

LOSSES=(
  #only-position
  #only-quaternion
  #quaternion-angle-homoscedastic
  quaternion-error-homoscedastic
  #naive-homoscedastic
  #naive-weighted
  #quaternion-error-weighted
  #quaternion-angle-weighted
)

if [[ "${MODE}" == "finetune" ]]; then
  feature_type=finetune
elif [[ "${MODE}" == "initial" ]]; then
  feature_type=cnn  
fi

for i in `seq $ITERS`; do
  for scene in "${SCENES[@]}"; do
    for topmodeltype in "${TOP_MODEL_TYPES[@]}"; do
      for config in "${CONFIGS[@]}"; do 

        IFS=',' read arch dataset batch_size hyperparams <<< "${config}"          
        scene_dir="${DATASET_DIR}/${scene}"
        echo 'Processing scene:' "${scene_dir}"

        train_seq_label_dirs=()
        valid_seq_label_dirs=()

        train_seq_feature_dirs=()
        valid_seq_feature_dirs=()
        
        if [[ "${topmodeltype}" == "regressor" ]] || [[ "${topmodeltype}" == "spatial-lstm" ]]; then

          for seq in ${scene_dir}/train/seq-*/; do
            train_seq_label_dirs+=("${seq}")
            train_seq_feature_dirs+=("${seq}/extracted_features/${arch}/${dataset}/${feature_type}_features.npy")
          done

          for seq in ${scene_dir}/test/seq-*/; do
            valid_seq_label_dirs+=("${seq}")
            valid_seq_feature_dirs+=("${seq}/extracted_features/${arch}/${dataset}/${feature_type}_features.npy")
          done

        elif [[ "${topmodeltype}" == "stateful-lstm" ]] || [[ "${topmodeltype}" == "standard-lstm" ]]; then
          
          lstm_type=$(echo "${topmodeltype}" | cut -d- -f1)

          prefix="${scene_dir}/extracted_sequences/${arch}/${dataset}/${lstm_type}/${SEQ_LEN}/"

          train_seq_feature_dirs+=("${prefix}/${lstm_type}_${feature_type}_train_features_seqs.npy")
          train_seq_label_dirs+=("${prefix}/${lstm_type}_${feature_type}_train_labels_seqs.npy")
          
          valid_seq_feature_dirs+=("${prefix}/${lstm_type}_${feature_type}_val_features_seqs.npy")
          valid_seq_label_dirs+=("${prefix}/${lstm_type}_${feature_type}_val_labels_seqs.npy")
        
        fi
        
        for loss in "${LOSSES[@]}"; do     

          output_dir="${OUTPUT_DIR}/${DATE}/${scene}/${topmodeltype}_${loss}_${arch}_${dataset}_seqlen=${SEQ_LEN}/"
          mkdir -p "${output_dir}"

          python -m "${MODULE}" \
            -tl "${train_seq_label_dirs[@]}" \
            -tf "${train_seq_feature_dirs[@]}" \
            -vl "${valid_seq_label_dirs[@]}" \
            -vf "${valid_seq_feature_dirs[@]}" \
            -o "${output_dir}" \
            --mode "${MODE}" \
            --top-model-type "${topmodeltype}" \
            --loss "${loss}" \
            --hyperparam-config "${HYPERPARAMS_MODULES}.${hyperparams}" \
            --finetuning-model-arch "${arch}" \
            --finetuning-model-dataset "${dataset}" \
            -i 1 \
            --epochs "${EPOCHS}" \
            --batch-size "${batch_size}" \
            --save-period 1 \
            --seq-len "${SEQ_LEN}" \
            --model-weights "${MODEL_WEIGHTS}"
        done
      done
    done
  done
done

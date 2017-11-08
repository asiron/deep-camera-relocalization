#!/bin/bash

MODULE="pose_regression.train"

DATE=`date +"%m_%d_%Y--%H-%M-%S"`

MODE=finetune
#MODE=initial

#WING_DATASET=wing
WING_DATASET=wing-5

DATASETS="/media/labuser/Storage/arg-00/datasets"
DATASET_DIR="${DATASETS}/${WING_DATASET}"
OUTPUT_DIR="/media/labuser/Flight_data1/maciej-cnn-${WING_DATASET}-results-${MODE}"

BATCH_SIZE=20
SEQ_LEN=5

HYPERPARAM_CONFIG="pose_regression.configs.hyperparam_finetune"

TOP_MODEL_TYPES=(
  #stateful-regressor-lstm
  #stateful-lstm
  standard-lstm
  #regressor
  #spatial-lstm
)

ITERS=400
EPOCHS=100


NETS=(
  #googlenet,imagenet
  #googlenet,places365
  #inception_resnet_v2,imagenet
  vgg16,hybrid1365
)

TOP_MODEL_WEIGHTS=

LOSSES=(
  #only_position
  #only_quaternion
  #quaternion_angle_homoscedastic
  quaternion_error_homoscedastic
  #naive_homoscedastic
  #naive_weighted
  #quaternion_error_weighted
  #quaternion_angle_weighted
)

if [[ "${MODE}" == "finetune" ]]; then
  feature_type=finetune
elif [[ "${MODE}" == "initial" ]]; then
  feature_type=cnn  
fi

for i in `seq $ITERS`; do
  for topmodeltype in "${TOP_MODEL_TYPES[@]}"; do
    for net in "${NETS[@]}"; do 
      IFS=',' read arch dataset <<< "${net}"
            
      train_seq_label_dirs=()
      valid_seq_label_dirs=() 

      train_seq_feature_dirs=()
      valid_seq_feature_dirs=() 
      
      if [[ "${topmodeltype}" == "regressor" ]] || [[ "${topmodeltype}" == "spatial-lstm" ]]; then

        for seq in ${DATASET_DIR}/position_*/train/seq_*; do
          train_seq_label_dirs+=("${seq}/labels")
          train_seq_feature_dirs+=("${seq}/extracted_features/${arch}/${dataset}/${feature_type}_features.npy")
        done

        for seq in ${DATASET_DIR}/position_*/test/seq_*; do
          valid_seq_label_dirs+=("${seq}/labels")
          valid_seq_feature_dirs+=("${seq}/extracted_features/${arch}/${dataset}/${feature_type}_features.npy")
        done  

      elif [[ "${topmodeltype}" == "stateful-lstm" ]] || [[ "${topmodeltype}" == "standard-lstm" ]]; then
        
        lstm_type=$(echo "${topmodeltype}" | cut -d- -f1)

        prefix="${DATASET_DIR}/extracted_sequences/${arch}/${dataset}/${lstm_type}/${SEQ_LEN}/"

        train_seq_feature_dirs+=("${prefix}/${lstm_type}_${feature_type}_train_features_seqs.npy")
        train_seq_label_dirs+=("${prefix}/${lstm_type}_${feature_type}_train_labels_seqs.npy")
        
        valid_seq_feature_dirs+=("${prefix}/${lstm_type}_${feature_type}_val_features_seqs.npy")
        valid_seq_label_dirs+=("${prefix}/${lstm_type}_${feature_type}_val_labels_seqs.npy")
             
      fi

      for loss in "${LOSSES[@]}"; do     

        output_dir="${OUTPUT_DIR}/${DATE}/${topmodeltype}-${loss}-${arch}-${dataset}/"
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
          --hyperparam-config "${HYPERPARAM_CONFIG}" \
          --finetuning-model-arch "${arch}" \
          --finetuning-model-dataset "${dataset}" \
          -i 1 \
          --epochs "${EPOCHS}" \
          --batch-size "${BATCH_SIZE}" \
          --save-period 1 \
          --seq-len "${SEQ_LEN}" \
          --top-model-weights "${TOP_MODEL_WEIGHTS}" 
      done
    done
  done
done

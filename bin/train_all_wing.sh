#!/bin/bash

MODULE="pose_regression.train"

#DATE=`date +"%m_%d_%Y--%H-%M-%S"`
DATE="11_23_2017--09-40-45"

MODE=finetune
#MODE=initial

#WING_DATASET=wing
WING_DATASET=wing-5

DATASETS="/media/labuser/Storage/arg-00/datasets"
DATASET_DIR="${DATASETS}/${WING_DATASET}"
OUTPUT_DIR="/media/labuser/Flight_data1/maciej-cnn-${WING_DATASET}-results-${MODE}"

SEQ_LEN=2
SUBSEQ_LEN=-1

# SEQ_LEN=77
# SUBSEQ_LEN=11

# SEQ_LEN=216
# SUBSEQ_LEN=18

HYPERPARAMS_MODULES="pose_regression.configs"

TOP_MODEL_TYPES=(
  #stateful-regressor-lstm
  #stateful-lstm
  standard-lstm
  #spatial-lstm
  #regressor
)

ITERS=100
EPOCHS=500

CONFIGS=(
  #googlenet,imagenet,64,settings.adam-no-modifier
  #googlenet,places365,64,settings.adam-no-modifier
  vgg16,hybrid1365,32,settings.adam-no-modifier
  #inception_resnet_v2,imagenet,20,settings.adam-no-modifier
)

MODEL_WEIGHTS=

LOSSES=(
  #only-position
  #only-quaternion
  #quaternion-angle-homoscedastic
  quaternion-error-homoscedastic
  naive-homoscedastic
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
  for topmodeltype in "${TOP_MODEL_TYPES[@]}"; do
    for config in "${CONFIGS[@]}"; do 
      IFS=',' read arch dataset batch_size hyperparams <<< "${config}"          
            
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
          --hyperparam-config "${HYPERPARAMS_MODULES}.${hyperparams}" \
          --finetuning-model-arch "${arch}" \
          --finetuning-model-dataset "${dataset}" \
          -i 1 \
          --epochs "${EPOCHS}" \
          --batch-size "${batch_size}" \
          --save-period 1 \
          --seq-len "${SEQ_LEN}" \
          --subseq-len "${SUBSEQ_LEN}" \
          --model-weights "${MODEL_WEIGHTS}" 
      done
    done
  done
done

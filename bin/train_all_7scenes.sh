#!/bin/bash

MODULE="pose_regression.train"

DATE=`date +"%m_%d_%Y--%H-%M-%S"`

MODE=finetune
#MODE=initial

RANDOM_CROP=false

DATASETS="/media/labuser/Storage/arg-00/datasets"

DATASET_DIR="${DATASETS}/7scenes"
OUTPUT_DIR="/media/labuser/Flight_data1/maciej-cnn-7scenes-results-${MODE}"

BATCH_SIZE=32
SEQ_LEN=4

ITERS=1000
EPOCHS=200

SCENES=(
  office
  #stairs
  #redkitchen
)

HYPERPARAM_CONFIG="pose_regression.configs.hyperparam_finetune"

TOP_MODEL_TYPES=(
  #stateful-regressor-lstm
  #stateful-lstm
  #standard-lstm
  #regressor
  spatial-lstm
)

NETS=(
  #googlenet,imagenet
  #googlenet,places365
  #inception_resnet_v2,imagenet
  vgg16,hybrid1365
)

#MODEL_WEIGHTS=/media/labuser/Flight_data1/maciej-cnn-7scenes-results-initial/10_21_2017--02-37-36/office/stateful-lstm-naive_homoscedastic-googlenet-places365/checkpoints/L1,decay=10,beta=132.7,lr=1.34e-04,dropout=0.128,l2_regu=0.056,lstm=1253/weights.0024-1.4515.hdf5
#MODEL_WEIGHTS=/media/labuser/Flight_data1/maciej-cnn-7scenes-results-initial/10_21_2017--02-37-36/office/stateful-lstm-naive_homoscedastic-googlenet-places365/checkpoints/L2,decay=10,beta=227.7,lr=1.38e-04,dropout=0.432,l2_regu=0.034,lstm=843/weights.0038--1.0286.hdf5

MODEL_WEIGHTS=/media/labuser/Flight_data1/maciej-cnn-7scenes-results-initial/11_07_2017--21-18-00/office/spatial-lstm_quaternion-error-weighted_vgg16_hybrid1365_seqlen=4/gamma=1,decay=5,beta=463.7,l_rate=1.27e-04,dropout=0.000,l2_regu=0.000,lstm_units=104,build=standard,r_act=tanh/checkpoints/weights.0006-15.2281.hdf5

LOSSES=(
  #only-position
  #only-quaternion
  #quaternion-angle-homoscedastic
  #quaternion-error-homoscedastic
  #naive-homoscedastic
  #naive-weighted
  quaternion-error-weighted
  #quaternion-angle-weighted
)

if [[ "${MODE}" == "finetune" ]]; then
  feature_type=finetune
elif [[ "${MODE}" == "initial" ]]; then
  feature_type=cnn  
fi

if [ "${RANDOM_CROP}" == true ]; then
  echo "RANDOM CROPS ENABLED!"
  random_crop_prefix="random_crop_"
  random_crops_flag="--random-crops"
else
  echo "RANDOM CROPS DISABLED!"
  random_crop_prefix=""
  random_crops_flag="--no-random-crops"
fi

for i in `seq $ITERS`; do
  for topmodeltype in "${TOP_MODEL_TYPES[@]}"; do
    for net in "${NETS[@]}"; do 
      IFS=',' read arch dataset <<< "${net}"
        
      for scene in "${SCENES[@]}"; do
        
        scene_dir="${DATASET_DIR}/${scene}"
        echo 'Processing scene:' "${scene_dir}"

        train_seq_label_dirs=()
        valid_seq_label_dirs=() 

        train_seq_feature_dirs=()
        valid_seq_feature_dirs=() 
        
        if [[ "${topmodeltype}" == "regressor" ]] || [[ "${topmodeltype}" == "spatial-lstm" ]]; then

          for seq in ${scene_dir}/train/seq-*/; do
            train_seq_label_dirs+=("${seq}")
            if [ "${RANDOM_CROP}" == false ]; then
              train_seq_feature_dirs+=("${seq}/extracted_features/${arch}/${dataset}/${feature_type}_features.npy")
            fi
          done

          for seq in ${scene_dir}/test/seq-*/; do
            valid_seq_label_dirs+=("${seq}")
            valid_seq_feature_dirs+=("${seq}/extracted_features/${arch}/${dataset}/${feature_type}_features.npy")
          done

          if [ "${RANDOM_CROP}" == true ]; then
            train_seq_feature_dirs+=("${scene_dir}/extracted_sequences/${arch}/${dataset}/random_crops_${feature_type}_features.npy")
          fi

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
            --hyperparam-config "${HYPERPARAM_CONFIG}" \
            --finetuning-model-arch "${arch}" \
            --finetuning-model-dataset "${dataset}" \
            -i 1 \
            --epochs "${EPOCHS}" \
            --batch-size "${BATCH_SIZE}" \
            --save-period 1 \
            --seq-len "${SEQ_LEN}" \
            --model-weights "${MODEL_WEIGHTS}" \
            "${random_crops_flag}"
          exit 1
        done
      done
    done
  done
done

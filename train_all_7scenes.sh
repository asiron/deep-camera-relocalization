#!/bin/bash

export CUDA_PATH=/usr/local/cuda-8.0/bin
export PATH=${CUDA_PATH}${PATH:+:${PATH}}
export LOCAL_CUDNN_PATH=~/cudnn/
export LD_LIBRARY_PATH=$LOCAL_CUDNN_PATH/cuda/lib64:$LD_LIBRARY_PATH

DATE=`date +"%m_%d_%Y--%H-%M-%S"`

DATASET_DIR="${HOME}/datasets/7scenes"

SCENES=(
  office
  stairs
  #redkitchen
)

#HYPERPARAM_CONFIG="configs.hyperparam_initial_config"
HYPERPARAM_CONFIG="configs.hyperparam_finetune"

ITERS=5
TOP_MODEL_TYPE=regressor

NETS=(
  #googlenet,imagenet
  googlenet,places365
  #inception_resnet_v2,imagenet
)

MODE=finetune
#FINETUNE_MODEL=/home/labuser/maciej/pose-regression/cnn/googlenet/places365/places365_last_inception.h5
#TOP_MODEL_WEIGHTS="/media/labuser/Flight_data/maciej-cnn-pose-regression-results/10_12_2017--00-12-18/redkitchen/initial-regressor-quaternion_homoscedastic-googlenet-imagenet/checkpoints/decay=240,beta=151.8,lr=5.12e-05,dropout=0.28,l2_regu=0.39/weights.0139-23.3887.hdf5"
#TOP_MODEL_WEIGHTS="/media/labuser/Flight_data/maciej-cnn-pose-regression-results--old-2/office/initial-regressor-naive_weighted-googlenet-places365/10_09_2017--23-53-43/checkpoints/decay=240,beta=47.6,lr=4.53e-05,dropout=0.10,l2_regu=0.13/weights.0289-6.0129.hdf5"

#TOP_MODEL_WEIGHTS="/media/labuser/Flight_data/maciej-cnn-pose-regression-results-initial/10_14_2017--03-14-15/office/initial-regressor-naive_weighted-inception_resnet_v2-imagenet/checkpoints/decay=50,beta=321.9,lr=4.75e-05,dropout=0.19,l2_regu=0.11/weights.0985-37.3448.hdf5"

#TOP_MODEL_WEIGHTS="/media/labuser/Flight_data/maciej-cnn-pose-regression-results--old-2/office/initial-regressor-naive_weighted-inception_resnet_v2-imagenet/10_09_2017--23-53-43/checkpoints/decay=160,beta=47.5,lr=9.74e-06,dropout=0.34,l2_regu=0.14/weights.0979-6.5907.hdf5"


TOP_MODEL_WEIGHTS="/media/labuser/Flight_data/maciej-cnn-pose-regression-results-initial/10_14_2017--20-13-31/office/initial-regressor-quaternion_homoscedastic-inception_resnet_v2-imagenet/checkpoints/L=1,decay=100,beta=306.4,lr=1.00e-04,dropout=0.50,l2_regu=0.01/weights.0143-0.1227.hdf5"

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
    
  for scene in "${SCENES[@]}"; do
    
    scene_dir="${DATASET_DIR}/${scene}"
    echo 'Processing scene:' "${scene_dir}"

    #train_seqs=$(cat "${scene_dir}/TrainSplit.txt" | tr '\n\r' ' ')
    #valid_seqs=$(cat "${scene_dir}/ValidSplit.txt" | tr '\n\r' ' ')
    #valid_seqs=$(cat "${scene_dir}/TestSplit.txt" | tr '\n\r' ' ')

    train_seq_label_dirs=()
    valid_seq_label_dirs=() 

    train_seq_feature_dirs=()
    valid_seq_feature_dirs=() 
    
    for seq in `ls "${scene_dir}/train"`; do
      train_seq_label_dirs+=("${scene_dir}/train/${seq}/")
      train_seq_feature_dirs+=("${scene_dir}/train/${seq}/extracted_features/${arch}/${dataset}/finetune_features.npy")
      #train_seq_feature_dirs+=(${scene_dir}/${seq}/extracted_features/${arch}/${dataset}/cnn_features.npy)
    done

    for seq in `ls "${scene_dir}/test"`; do
      valid_seq_label_dirs+=("${scene_dir}/test/${seq}/")
      valid_seq_feature_dirs+=("${scene_dir}/test/${seq}/extracted_features/${arch}/${dataset}/finetune_features.npy")
      #valid_seq_feature_dirs+=(${scene_dir}/${seq}/extracted_features/${arch}/${dataset}/cnn_features.npy)
    done  

    #for loss in naive_weighted quaternion_weighted; do 
    #for loss in naive_weighted; do 
    
    for loss in "${LOSSES[@]}"; do     
      OUTPUT_DIR="/media/labuser/Flight_data/maciej-cnn-pose-regression-results-finetune"
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
          --top-model-weights "${TOP_MODEL_WEIGHTS}" \
          --finetuning-model-arch "${arch}" \
          --finetuning-model-dataset "${dataset}" \
          -i 1 \
          --epochs 50 \
          --batch-size 64 \
          --save-period 1

      done
    done
  done
done

#!/bin/bash

MODULE=pose_regression.scripts.extract_features


DATASET_DIR="${DATASETS}/${WING_DATASET}"

function process_dataset {

  dataset=$1
  batch_size=$2

  meanfile="${dataset}/meanfiles/224/meanfile.npy"

  for position in $dataset/position_*/; do
    
    echo 'Processing : ' $position
    for seq in $position/t*/seq_*/; do

      echo -e '\tProcessing sequence:' $seq

      python -m "${MODULE}" -m googlenet -d places365 \
        --meanfile "${meanfile}" --batch-size "${batch_size}" \
        "${seq}/images" "${seq}/extracted_features/googlenet/places365"

      python -m "${MODULE}" -m googlenet -d imagenet \
        --meanfile "${meanfile}" --batch-size "${batch_size}" \
        "${seq}/images" "${seq}/extracted_features/googlenet/imagenet"

      python -m "${MODULE}" -m inception_resnet_v2 -d imagenet \
        --batch-size $(echo "${batch_size}/6" | bc) \
        "${seq}/images" "${seq}/extracted_features/inception_resnet_v2/imagenet"
    
      python -m "${MODULE}" -m vgg16 -d hybrid1365 \
        --meanfile "${meanfile}" \
        --batch-size $(echo "${batch_size}/6" | bc) \
        "${seq}/images" "${seq}/extracted_features/vgg16/hybrid1365"

    done
  done
}

DATASETS="/media/labuser/Storage/arg-00/datasets"

process_dataset "${DATASETS}/wing" 128
process_dataset "${DATASETS}/wing-5" 128

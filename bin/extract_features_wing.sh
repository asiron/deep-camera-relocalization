#!/bin/bash

MODULE=pose_regression.scripts.extract_features

WING_DATASET=wing
#WING_DATASET=wing-5

DATASETS="/media/labuser/Storage/arg-00/datasets"
DATASET_DIR="${DATASETS}/${WING_DATASET}"

MEANFILE="${DATASET_DIR}/meanfiles/224/meanfile.npy"

function process_dataset {
  dataset=$1
  batch_size=$2
  random_crops=$3

  for position in $DATASET_DIR/position_*/; do
    
    echo 'Processing : ' $position
    for seq in $position/$dataset/seq_*/; do

      echo -e '\tProcessing sequence:' $seq

      python -m "${MODULE}" -m googlenet -d places365 \
        --meanfile "${MEANFILE}" --batch-size "${batch_size}" \
        --random-crops "${random_crops}" \
        "${seq}/images" "${seq}/extracted_features/googlenet/places365"

      python -m "${MODULE}" -m googlenet -d imagenet \
        --meanfile "${MEANFILE}" --batch-size "${batch_size}" \
        --random-crops "${random_crops}" \
        "${seq}/images" "${seq}/extracted_features/googlenet/imagenet"

      python -m "${MODULE}" -m inception_resnet_v2 -d imagenet \
        --batch-size $(echo "${batch_size}/4" | bc) \
        --random-crops "${random_crops}" \
        "${seq}/images" "${seq}/extracted_features/inception_resnet_v2/imagenet"

    done
  done
}

process_dataset train 128 32
process_dataset train 128 0
process_dataset test 128 0

#!/bin/bash

MODULE=pose_regression.scripts.extract_features

DATASETS="/media/labuser/Storage/arg-00/datasets"
DATASET_DIR="${DATASETS}/7scenes"

function process_dataset {
  dataset=$1
  batch_size=$2
  random_crops=$3

  echo 'Processing dataset: ' $1
  for seq in $1/seq-*/; do

    pattern="^frame-[0-9]{6}\.color\.png$"
    echo -e '\tProcessing sequence:' $seq
    echo -e '\tUsing pattern: ' $pattern

    meanfile="$(dirname $dataset)/train/meanfiles/224/meanfile.npy"

    python -m "${MODULE}" -p $pattern -m googlenet -d places365 \
      --meanfile "${meanfile}" --batch-size "${batch_size}" \
      --random-crops "${random_crops}" \
      "${seq}" "${seq}/extracted_features/googlenet/places365"

    python -m "${MODULE}" -p $pattern -m googlenet -d imagenet \
      --meanfile "${meanfile}" --batch-size "${batch_size}" \
      --random-crops "${random_crops}" \
      "${seq}" "${seq}/extracted_features/googlenet/imagenet"

    # python -m "${MODULE}" -p $pattern -m inception_resnet_v2 -d imagenet \
    #   --batch-size $(echo "${batch_size}/4" | bc) --random-crops "${random_crops}" \
    #   "${seq}" "${seq}/extracted_features/inception_resnet_v2/imagenet"

  done
}

#process_dataset $DATASET_DIR/office/train 128 32
#process_dataset $DATASET_DIR/office/train 128 0
process_dataset $DATASET_DIR/office/test 128 0

#!/bin/bash

MODULE=pose_regression.scripts.extract_features

DATASETS="/media/labuser/Storage/arg-00/datasets"
DATASET_DIR="${DATASETS}/7scenes"

function process_dataset {
  dataset=$1
  batch_size=$2

  echo 'Processing dataset: ' $dataset
  for seq in $dataset/t*/seq-*/; do

    pattern="^frame-[0-9]{6}\.color\.png$"
    echo -e '\tProcessing sequence:' $seq
    echo -e '\tUsing pattern: ' $pattern

    meanfile="$dataset/train/meanfiles/224/meanfile.npy"

    # python -m "${MODULE}" -p $pattern -m googlenet -d places365 \
    #   --meanfile "${meanfile}" --batch-size "${batch_size}" \
    #   "${seq}" "${seq}/extracted_features/googlenet/places365"

    # python -m "${MODULE}" -p $pattern -m googlenet -d imagenet \
    #   --meanfile "${meanfile}" --batch-size "${batch_size}" \
    #   "${seq}" "${seq}/extracted_features/googlenet/imagenet"

    # python -m "${MODULE}" -p $pattern -m inception_resnet_v2 -d imagenet \
    #   --batch-size $(echo "${batch_size}/4" | bc) \
    #   "${seq}" "${seq}/extracted_features/inception_resnet_v2/imagenet"

    python -m "${MODULE}" -p $pattern -m vgg16 -d hybrid1365 \
      --meanfile "${meanfile}" \
      --batch-size $(echo "${batch_size}/4" | bc) \
      "${seq}" "${seq}/extracted_features/vgg16/hybrid1365"

  done
}

process_dataset $DATASET_DIR/office 80
# process_dataset $DATASET_DIR/chess 128
# process_dataset $DATASET_DIR/fire 128
# process_dataset $DATASET_DIR/heads 128
# process_dataset $DATASET_DIR/pumpkin 128
# process_dataset $DATASET_DIR/redkitchen 128
# process_dataset $DATASET_DIR/stairs 128

#!/bin/bash

DATASET_DIR="${HOME}/datasets/wing"

for position in $DATASET_DIR/position_*/; do
  
  echo 'Processing : ' $position
  for seq in $position/t*/seq_*/; do

    echo -e '\tProcessing sequence:' $seq

    python extract_features.py -m googlenet -d places365 --batch-size 128 \
     --meanfile $DATASET_DIR/meanfiles/224/meanfile.npy \
      "${seq}/images" "${seq}/extracted_features/googlenet/places365"

    # python extract_features.py -m inception_resnet_v2 -d imagenet \
    #   "${seq}/images" "${seq}/extracted_features/inception_resnet_v2/imagenet"

    # python extract_features.py -m googlenet -d imagenet \
    #   "${seq}/images" "${seq}/extracted_features/googlenet/imagenet"


  done
done

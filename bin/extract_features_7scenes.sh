#!/bin/bash

DATASET_DIR="${HOME}/datasets/7scenes"

#for dataset in $DATASET_DIR/*/; do
for dataset in $DATASET_DIR/office/; do
  
  echo 'Processing dataset: ' $dataset
  for seq in $dataset/t*/seq-*/; do

    pattern="^frame-[0-9]{6}\.color\.png$"
    echo -e '\tProcessing sequence:' $seq
    echo -e '\tUsing pattern: ' $pattern

    # python extract_features.py -p $pattern -m googlenet -d places205 --batch-size 128 \
    #   "${seq}" "${seq}/extracted_features/googlenet/places205"

    # python extract_features.py -p $pattern -m inception_resnet_v2 -d imagenet \
    #   "${seq}" "${seq}/extracted_features/inception_resnet_v2/imagenet"

    # python extract_features.py -p $pattern -m googlenet -d imagenet \
    #   "${seq}" "${seq}/extracted_features/googlenet/imagenet"

    python extract_features.py -p $pattern -m googlenet -d places365 --batch-size 128\
      "${seq}" "${seq}/extracted_features/googlenet/places365"

  done
done

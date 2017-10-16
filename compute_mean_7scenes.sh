#!/bin/bash

DATASET_DIR="${HOME}/datasets/7scenes"

TMP_FILE="/tmp/image_files.txt"

#for dataset in $DATASET_DIR/*/; do
for dataset in $DATASET_DIR/office/train; do
  
    files=$(find $dataset -regextype sed -regex ".*frame-[0-9]\{6,6\}\.color\.png$" | tr '\n' ' ')
    echo "${files}" > "${TMP_FILE}"
    
    python compute_mean.py --resize 224x224 --batch-size 100 \
      -i "${TMP_FILE}" -o cnn/googlenet/places205/

    rm "${TMP_FILE}"

    # python extract_features.py -p $pattern -m inception_resnet_v2 -d imagenet \
    #   "${seq}" "${seq}/extracted_features/inception_resnet_v2/imagenet"

    # python extract_features.py -p $pattern -m googlenet -d imagenet \
    #   "${seq}" "${seq}/extracted_features/googlenet/imagenet"

    # python extract_features.py -p $pattern -m googlenet -d places365 \
    #   "${seq}" "${seq}/extracted_features/googlenet/places365"

done

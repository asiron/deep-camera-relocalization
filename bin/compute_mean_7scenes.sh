#!/bin/bash

MODULE="pose_regression.scripts.compute_mean"

DATASET_DIR="${HOME}/datasets/7scenes"

TFFILE=$(mktemp /tmp/foo.XXXXXXXXX)

for dataset in $DATASET_DIR/*/; do

    echo "Processing dataset: " $dataset
  
    files=$(find $dataset -regextype sed -regex ".*frame-[0-9]\{6,6\}\.color\.png$" | tr '\n' ' ')
    echo "${files}" > "${TFFILE}"
   
    output_dir="${dataset}/meanfiles"
    mkdir -p "${output_dir}/224" "${output_dir}/299"

    echo "Computing mean with size 224x224"
    python -m "${MODULE}" \
      --resize 224x224 \
      --batch-size 100 \
      -i "${TFFILE}" \
      -o "${output_dir}/224"

    echo "Computing mean with size 299x299"
    python -m "${MODULE}" \
      --resize 299x299 \
      --batch-size 100 \
      -i "${TFFILE}" \
      -o "${output_dir}/299"

    rm "${TFFILE}"

done

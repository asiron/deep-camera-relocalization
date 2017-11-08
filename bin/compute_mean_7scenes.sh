#!/bin/bash

MODULE="pose_regression.scripts.compute_mean"


DATASETS="/media/labuser/Storage/arg-00/datasets"
DATASET_DIR="${DATASETS}/7scenes"


for dataset in $DATASET_DIR/*/train; do

    echo "Processing dataset: " $dataset
    
    tffile=$(mktemp /tmp/foo.XXXXXXXXX)
  
    files=$(find $dataset -regextype sed -regex ".*frame-[0-9]\{6,6\}\.color\.png$" | tr '\n' ' ')
    echo "${files}" > "${tffile}"
   
    output_dir="${dataset}/meanfiles"
    mkdir -p "${output_dir}/224" "${output_dir}/299"

    echo "Computing mean with size 224x224"
    python -m "${MODULE}" \
      --resize 224x224 \
      --batch-size 128 \
      -i "${tffile}" \
      -o "${output_dir}/224"

    rm "${tffile}"

done

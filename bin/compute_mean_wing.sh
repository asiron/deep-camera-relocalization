#!/bin/bash

MODULE="pose_regression.scripts.compute_mean"

DATASET_DIR="${HOME}/datasets/wing"

TFFILE=$(mktemp /tmp/foo.XXXXXXXXX)
FILES=$(find $DATASET_DIR -regextype sed -regex ".*/train/.*/image_[0-9]\{5,5\}\.png$" | tr '\n' ' ')
echo "${FILES}" > "${TFFILE}"

OUTPUT_DIR=$DATASET_DIR/meanfiles/
mkdir -p "${OUTPUT_DIR}/224" "${OUTPUT_DIR}/299"

python -m "${MODULE}" \
  --resize 224x224 \
  --batch-size 100 \
  -i "${TFFILE}" \
  -o "${OUTPUT_DIR}/224"

python -m "${MODULE}" \
  --resize 299x299 \
  --batch-size 100 \
  -i "${TFFILE}" \
  -o "${OUTPUT_DIR}/299"

rm "${TFFILE}"

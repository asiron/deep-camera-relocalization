#!/bin/bash

DATASET_DIR="${HOME}/datasets/wing"
TMP_FILE="/tmp/image_files.txt"

  

FILES=$(find $DATASET_DIR -regextype sed -regex ".*/train/.*/image_[0-9]\{5,5\}\.png$" | tr '\n' ' ')

echo "${FILES}" > "${TMP_FILE}"

OUTPUT_DIR=$DATASET_DIR/meanfiles/

mkdir -p $OUTPUT_DIR/224 $OUTPUT_DIR/299

python compute_mean.py --resize 224x224 --batch-size 100 \
  -i "${TMP_FILE}" -o $OUTPUT_DIR/224

python compute_mean.py --resize 299x299 --batch-size 100 \
  -i "${TMP_FILE}" -o $OUTPUT_DIR/299

rm "${TMP_FILE}"


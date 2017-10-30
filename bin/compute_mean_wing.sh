#!/bin/bash

MODULE="pose_regression.scripts.compute_mean"


WING_DATASET=wing
#WING_DATASET=wing-5

DATASETS="/media/labuser/Storage/arg-00/datasets"
DATASET_DIR="${DATASETS}/${WING_DATASET}"

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

rm "${TFFILE}"

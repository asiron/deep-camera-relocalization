#!/bin/bash

JOBS=10
MODULE="pose_regression.scripts.convert_pose_file"

DATASETS="/media/labuser/Storage/arg-00/datasets"
DATASET_DIR="${DATASETS}/7scenes/office"

files=$(find "${DATASET_DIR}" -regex ".*frame-[0-9]*\.pose\.txt$")
count=$(echo "${files}" | wc -l)

TFFILE=$(mktemp /tmp/foo.XXXXXXXXX)
echo "${files}" > "${TFFILE}"

pv "${TFFILE}" | xargs -l -n1 -P $JOBS python -m "${MODULE}"

rm "${TFFILE}"

#!/bin/bash

JOBS=12
MODULE="pose_regression.scripts.convert_pose_file"
DATASET_DIR="${HOME}/datasets/7scenes"

files=$(find "${DATASET_DIR}" -regex ".*frame-[0-9]*\.pose\.txt$")
count=$(echo "${files}" | wc -l)

TFFILE=$(mktemp /tmp/foo.XXXXXXXXX)
echo "${files}" > "${TFFILE}"

pv "${TFFILE}" | xargs -l -n1 -P $JOBS python -m "${MODULE}"

rm "${TFFILE}"

#!/bin/bash

MODULE="pose_regression.scripts.compute_mean"

DATASETS="/media/labuser/Storage/arg-00/datasets"

function compute_mean {

	dataset=$1

	echo "Computing mean for dataset ${dataset}"

	tffile=$(mktemp /tmp/foo.XXXXXXXXX)
	files=$(find $dataset -regextype sed -regex ".*/train/.*/image_[0-9]\{5,5\}\.png$" | tr '\n' ' ')
	echo "${files}" > "${tffile}"

	output_dir=$dataset/meanfiles/
	mkdir -p "${output_dir}/224" "${output_dir}/299"

	python -m "${MODULE}" \
	  --resize 224x224 \
	  --batch-size 100 \
	  -i "${tffile}" \
	  -o "${output_dir}/224"

	rm "${tffile}"

}

compute_mean "${DATASETS}/wing"
compute_mean "${DATASETS}/wing-5"

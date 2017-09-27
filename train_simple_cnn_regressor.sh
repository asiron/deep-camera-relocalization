#!/bin/bash

DATASET_DIR=.

FEATURES=(
	"${DATASET_DIR}/extracted-inception-v3-position00/features.npy"
	"${DATASET_DIR}/extracted-inception-v3-position01/features.npy"
)

LABELS=(
	"../../datasets/wing/position00/seq00/labels"
	"../../datasets/wing/position01/seq00/labels"
)

OUTPUT_DIR="test-run-`date +"%m_%d_%Y--%H-%M-%S"`"

python train_simple_cnn_regressor.py \
	-l "${LABELS[@]}" \
	-f "${FEATURES[@]}" \
	-o "${OUTPUT_DIR}" \
	-i 200
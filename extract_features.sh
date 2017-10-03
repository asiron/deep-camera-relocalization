#!/bin/bash

DATASET_DIR="../../datasets/wing/"

SEQ00="${DATASET_DIR}/position00/seq00"
SEQ01="${DATASET_DIR}/position01/seq01"

python extract_features.py -m inception_resnet_v2 -d imagenet \
  "${SEQ00}/images" "${SEQ00}/extracted_features/inception_resnet_v2/imagenet"
python extract_features.py -m inception_resnet_v2 -d imagenet \
  "${SEQ01}/images" "${SEQ01}/extracted_features/inception_resnet_v2/imagenet"

python extract_features.py -m googlenet -d imagenet \
  "${SEQ00}/images" "${SEQ00}/extracted_features/googlenet/imagenet"
python extract_features.py -m googlenet -d imagenet \
  "${SEQ01}/images" "${SEQ01}/extracted_features/googlenet/imagenet"

python extract_features.py -m googlenet -d places365 \
  "${SEQ00}/images" "${SEQ00}/extracted_features/googlenet/places365"
python extract_features.py -m googlenet -d places365 \
  "${SEQ01}/images" "${SEQ01}/extracted_features/googlenet/places365"

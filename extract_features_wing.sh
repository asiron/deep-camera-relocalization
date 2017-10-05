#!/bin/bash

DATASET_DIR="../../datasets/wing/"

SET00="${DATASET_DIR}/position00/seq00"
SET01="${DATASET_DIR}/position01/seq00"


python extract_features.py -m googlenet -d places365 \
  "${SET01}/images" "${SET01}/extracted_features/googlenet/places365"
python extract_features.py -m googlenet -d places365 \
 "${SET00}/images" "${SET00}/extracted_features/googlenet/places365"

python extract_features.py -m googlenet -d imagenet \
  "${SET00}/images" "${SET00}/extracted_features/googlenet/imagenet"
python extract_features.py -m googlenet -d imagenet \
  "${SET01}/images" "${SET01}/extracted_features/googlenet/imagenet"

# python extract_features.py -m inception_resnet_v2 -d imagenet \
#   "${SET00}/images" "${SET00}/extracted_features/inception_resnet_v2/imagenet"
# python extract_features.py -m inception_resnet_v2 -d imagenet \
#   "${SET01}/images" "${SET01}/extracted_features/inception_resnet_v2/imagenet"

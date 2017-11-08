#!/bin/bash

MODULE="pose_regression.saliency"

DATE=`date +"%m_%d_%Y--%H-%M-%S"`

DATASETS="/media/labuser/Storage/arg-00/datasets"
DATASET_DIR="${DATASETS}/7scenes"

OUTPUT_DIR="/media/labuser/Flight_data1/maciej-cnn-7scenes-saliency"

BATCH_SIZE=32

MODEL_WEIGHTS=/media/labuser/Flight_data1/maciej-cnn-7scenes-results-finetune/11_07_2017--21-31-03/office/spatial-lstm_quaternion-error-weighted_vgg16_hybrid1365_seqlen=4/gamma=1,decay=5,beta=178.5,l_rate=1.27e-04,dropout=0.000,l2_regu=0.000,lstm_units=104,build=standard,r_act=tanh/checkpoints/weights.0001-4.3816.hdf5

scene_dir="${DATASET_DIR}/${scene}"
echo 'Processing scene:' "${scene_dir}"

output_dir="${OUTPUT_DIR}"
mkdir -p "${output_dir}"

python -m "${MODULE}" \
  --images aaa aaa \
  --output "${output_dir}" \
  --batch-size "${BATCH_SIZE}" \
  --model-weights "${MODEL_WEIGHTS}"
exit 1
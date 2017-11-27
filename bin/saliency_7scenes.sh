#!/bin/bash

MODULE="pose_regression.saliency"

DATE=`date +"%m_%d_%Y--%H-%M-%S"`

DATASETS="/media/labuser/Storage/arg-00/datasets"
DATASET_DIR="${DATASETS}/7scenes"

OUTPUT_DIR="/media/labuser/Flight_data1/maciej-cnn-7scenes-saliency"

BATCH_SIZE=32

MODEL_WEIGHTS=/media/labuser/Flight_data1/maciej-cnn-7scenes-results-finetune/11_08_2017--22-53-15/office/spatial-lstm_quaternion-error-homoscedastic_vgg16_hybrid1365_seqlen=4/gamma=1,decay=5,beta=120.7,l_rate=1.00e-04,dropout=0.00001,l2_regu=0.00001,lstm_units=104,build=standard,r_act=tanh/checkpoints/weights.0034--3.2954.hdf5

scene_dir="${DATASET_DIR}/${scene}"
echo 'Processing scene:' "${scene_dir}"

output_dir="${OUTPUT_DIR}"
mkdir -p "${output_dir}"

IMAGES=(
  /media/labuser/Storage/arg-00/datasets/7scenes/office/test/seq-06/frame-000732.color.png
)

python -m "${MODULE}" \
  --images "${IMAGES[@]}" \
  --output "${output_dir}" \
  --batch-size "${BATCH_SIZE}" \
  --model-weights "${MODEL_WEIGHTS}"
exit 1
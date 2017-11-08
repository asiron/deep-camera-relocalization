#!/bin/bash

echo "Computing means"

echo "Computing means for 7scenes"
bash bin/compute_mean_7scenes.sh 2>&1 | tee -a 7scenes-datasets-mean.log

echo "Computing means for wing datasets"
bash bin/compute_mean_wing.sh 2>&1 | tee -a wing-datasets-mean.log


echo "Extracting features"

echo "Extracting features for 7scenes"
bash bin/extract_features_7scenes.sh 2>&1 | tee -a 7scenes-datasets-features.log

echo "Extracting features for wing datasets"
bash bin/extract_features_wing.sh 2>&1 | tee -a wing-datasets-features.log


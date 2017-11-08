from __future__ import print_function

import argparse
import numpy as np

import keras.backend as K

from keras.models import Model
from keras.layers import (Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D)

from vgg16 import VGG16

def create_vgg16(weights_path, model_output_path):
  ''' supports only NHWC format'''
 
  K.set_image_data_format('channels_last') 

  input = Input(shape=VGG16.INPUT_SHAPE, name='cnn_input')
  
  conv1_1 = Conv2D(64, (3,3), strides=(1,1), activation='relu', name='conv1_1', padding='same')(input)
  conv1_2 = Conv2D(64, (3,3), strides=(1,1), activation='relu', name='conv1_2', padding='same')(conv1_1)
  pool1   = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='pool1', padding='same')(conv1_2)

  conv2_1 = Conv2D(128, (3,3), strides=(1,1), activation='relu', name='conv2_1', padding='same')(pool1)
  conv2_2 = Conv2D(128, (3,3), strides=(1,1), activation='relu', name='conv2_2', padding='same')(conv2_1)
  pool2   = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='pool2', padding='same')(conv2_2)

  conv3_1 = Conv2D(256, (3,3), strides=(1,1), activation='relu', name='conv3_1', padding='same')(pool2)
  conv3_2 = Conv2D(256, (3,3), strides=(1,1), activation='relu', name='conv3_2', padding='same')(conv3_1)
  conv3_3 = Conv2D(256, (3,3), strides=(1,1), activation='relu', name='conv3_3', padding='same')(conv3_2)
  pool3   = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='pool3', padding='same')(conv3_3)

  conv4_1 = Conv2D(512, (3,3), strides=(1,1), activation='relu', name='conv4_1', padding='same')(pool3)
  conv4_2 = Conv2D(512, (3,3), strides=(1,1), activation='relu', name='conv4_2', padding='same')(conv4_1)
  conv4_3 = Conv2D(512, (3,3), strides=(1,1), activation='relu', name='conv4_3', padding='same')(conv4_2)
  pool4   = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='pool4', padding='same')(conv4_3)

  fintuning_extract_input_shape = pool4._keras_shape
  fintuning_extract_input = Input(shape=fintuning_extract_input_shape[1:], name='finetuning_extract_layer')

  conv5_1 = Conv2D(512, (3,3), strides=(1,1), activation='relu', name='conv5_1', padding='same')(fintuning_extract_input)
  conv5_2 = Conv2D(512, (3,3), strides=(1,1), activation='relu', name='conv5_2', padding='same')(conv5_1)
  conv5_3 = Conv2D(512, (3,3), strides=(1,1), activation='relu', name='conv5_3', padding='same')(conv5_2)
  pool5   = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='pool5', padding='same')(conv5_3)
  
  output = GlobalAveragePooling2D()(pool5)

  vgg16_base = Model(inputs=input, outputs=pool4)
  vgg16_finetuning = Model(inputs=fintuning_extract_input, outputs=output)
  
  weights = np.load(weights_path)

  def load_weights(model, weights):
    gen = (l for l in model.layers if l.name in weights.item().keys())
    for l in gen:
      ws = weights.item()[l.name]
      l.set_weights([ws['weights'], ws['biases']])

  load_weights(vgg16_base, weights)
  load_weights(vgg16_finetuning, weights)

  vgg16_base.save('{}_base.h5'.format(model_output_path))
  vgg16_finetuning.save('{}_finetuning.h5'.format(model_output_path))

  print('VGG-16 base model:')
  vgg16_base.summary()

  print('VGG-16 finetuning model:')
  vgg16_finetuning.summary()

  print('Done!')

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('-w', '--weights', required=True, 
    help='Path a NumPy array with weights (params) of GoogLeNet')
  parser.add_argument('-m', '--model-output', required=True,
    help='Path to an output model file')
  args = parser.parse_args()

  create_vgg16(args.weights, args.model_output)

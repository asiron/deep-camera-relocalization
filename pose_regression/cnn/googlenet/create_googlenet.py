from __future__ import print_function

import argparse
import numpy as np

import tensorflow as tf
import keras.backend as K

from keras.models import Model
from keras.layers import (
  Input, Dense, Conv2D, MaxPooling2D, 
  AveragePooling2D, Dropout, Flatten, Activation, 
  Lambda, GlobalAveragePooling2D)
from keras.layers.merge import Concatenate
from keras.regularizers import l2

from googlenet_layers import LRN

def create_googlenet(weights_path, model_output_path):
  # creates GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
  
  '''
    channels last, because current implementation of LRN in TensorFlow 
    supports only NHWC
  '''
  K.set_image_data_format('channels_last') 

  input = Input(shape=(224, 224, 3), name='cnn_input')
  
  conv1_7x7_s2 = Conv2D(64, (7,7), strides=(2,2), padding='same',activation='relu',name='conv1_7x7_s2', kernel_regularizer=l2(0.0002))(input)
  pool1_3x3_s2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same',name='pool1_3x3_s2')(conv1_7x7_s2)
  pool1_norm1 = LRN(name='pool1_norm1')(pool1_3x3_s2)

  conv2_3x3_reduce = Conv2D(64, (1,1), padding='same',activation='relu',name='conv2_3x3_reduce', kernel_regularizer=l2(0.0002))(pool1_norm1)
  conv2_3x3 = Conv2D(192, (3,3), padding='same',activation='relu',name='conv2_3x3', kernel_regularizer=l2(0.0002))(conv2_3x3_reduce)  
  conv2_norm2 = LRN(name='conv2_norm2')(conv2_3x3)

  pool2_3x3_s2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same',name='pool2_3x3_s2')(conv2_norm2)
  
  inception_3a_1x1 = Conv2D(64, (1,1), padding='same',activation='relu',name='inception_3a_1x1', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
  inception_3a_3x3_reduce = Conv2D(96, (1,1), padding='same',activation='relu',name='inception_3a_3x3_reduce', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
  inception_3a_3x3 = Conv2D(128, (3,3), padding='same',activation='relu',name='inception_3a_3x3', kernel_regularizer=l2(0.0002))(inception_3a_3x3_reduce)
  inception_3a_5x5_reduce = Conv2D(16, (1,1), padding='same',activation='relu',name='inception_3a_5x5_reduce', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
  inception_3a_5x5 = Conv2D(32, (5,5), padding='same',activation='relu',name='inception_3a_5x5', kernel_regularizer=l2(0.0002))(inception_3a_5x5_reduce)
  inception_3a_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='inception_3a_pool')(pool2_3x3_s2)
  inception_3a_pool_proj = Conv2D(32, (1,1), padding='same',activation='relu',name='inception_3a_pool_proj', kernel_regularizer=l2(0.0002))(inception_3a_pool)
  inception_3a_output = Concatenate(axis=-1,name='inception_3a_output')([inception_3a_1x1,inception_3a_3x3,inception_3a_5x5,inception_3a_pool_proj])
  
  inception_3b_1x1 = Conv2D(128, (1,1), padding='same',activation='relu',name='inception_3b_1x1', kernel_regularizer=l2(0.0002))(inception_3a_output)  
  inception_3b_3x3_reduce = Conv2D(128, (1,1), padding='same',activation='relu',name='inception_3b_3x3_reduce', kernel_regularizer=l2(0.0002))(inception_3a_output)  
  inception_3b_3x3 = Conv2D(192, (3,3), padding='same',activation='relu',name='inception_3b_3x3', kernel_regularizer=l2(0.0002))(inception_3b_3x3_reduce)
  inception_3b_5x5_reduce = Conv2D(32, (1,1), padding='same',activation='relu',name='inception_3b_5x5_reduce', kernel_regularizer=l2(0.0002))(inception_3a_output) 
  inception_3b_5x5 = Conv2D(96, (5,5), padding='same',activation='relu',name='inception_3b_5x5', kernel_regularizer=l2(0.0002))(inception_3b_5x5_reduce)
  inception_3b_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='inception_3b_pool')(inception_3a_output)    
  inception_3b_pool_proj = Conv2D(64, (1,1), padding='same',activation='relu',name='inception_3b_pool_proj', kernel_regularizer=l2(0.0002))(inception_3b_pool)    
  inception_3b_output = Concatenate(axis=-1,name='inception_3b_output')([inception_3b_1x1,inception_3b_3x3,inception_3b_5x5,inception_3b_pool_proj])    
  pool3_3x3_s2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same',name='pool3_3x3_s2')(inception_3b_output)
  
  inception_4a_1x1 = Conv2D(192, (1,1), padding='same',activation='relu',name='inception_4a_1x1', kernel_regularizer=l2(0.0002))(pool3_3x3_s2)    
  inception_4a_3x3_reduce = Conv2D(96, (1,1), padding='same',activation='relu',name='inception_4a_3x3_reduce', kernel_regularizer=l2(0.0002))(pool3_3x3_s2)    
  inception_4a_3x3 = Conv2D(208, (3,3), padding='same',activation='relu',name='inception_4a_3x3', kernel_regularizer=l2(0.0002))(inception_4a_3x3_reduce)    
  inception_4a_5x5_reduce = Conv2D(16, (1,1), padding='same',activation='relu',name='inception_4a_5x5_reduce', kernel_regularizer=l2(0.0002))(pool3_3x3_s2)    
  inception_4a_5x5 = Conv2D(48, (5,5), padding='same',activation='relu',name='inception_4a_5x5', kernel_regularizer=l2(0.0002))(inception_4a_5x5_reduce)    
  inception_4a_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='inception_4a_pool')(pool3_3x3_s2)    
  inception_4a_pool_proj = Conv2D(64, (1,1), padding='same',activation='relu',name='inception_4a_pool_proj', kernel_regularizer=l2(0.0002))(inception_4a_pool)    
  inception_4a_output = Concatenate(axis=-1,name='inception_4a_output')([inception_4a_1x1,inception_4a_3x3,inception_4a_5x5,inception_4a_pool_proj])

  inception_4b_1x1 = Conv2D(160, (1,1), padding='same',activation='relu',name='inception_4b_1x1', kernel_regularizer=l2(0.0002))(inception_4a_output)    
  inception_4b_3x3_reduce = Conv2D(112, (1,1), padding='same',activation='relu',name='inception_4b_3x3_reduce', kernel_regularizer=l2(0.0002))(inception_4a_output)   
  inception_4b_3x3 = Conv2D(224, (3,3), padding='same',activation='relu',name='inception_4b_3x3', kernel_regularizer=l2(0.0002))(inception_4b_3x3_reduce)    
  inception_4b_5x5_reduce = Conv2D(24, (1,1), padding='same',activation='relu',name='inception_4b_5x5_reduce', kernel_regularizer=l2(0.0002))(inception_4a_output)    
  inception_4b_5x5 = Conv2D(64, (5,5), padding='same',activation='relu',name='inception_4b_5x5', kernel_regularizer=l2(0.0002))(inception_4b_5x5_reduce)    
  inception_4b_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='inception_4b_pool')(inception_4a_output)    
  inception_4b_pool_proj = Conv2D(64, (1,1), padding='same',activation='relu',name='inception_4b_pool_proj', kernel_regularizer=l2(0.0002))(inception_4b_pool)    
  inception_4b_output = Concatenate(axis=-1,name='inception_4b_output')([inception_4b_1x1,inception_4b_3x3,inception_4b_5x5,inception_4b_pool_proj])
  
  inception_4c_1x1 = Conv2D(128, (1,1), padding='same',activation='relu',name='inception_4c_1x1', kernel_regularizer=l2(0.0002))(inception_4b_output)    
  inception_4c_3x3_reduce = Conv2D(128, (1,1), padding='same',activation='relu',name='inception_4c_3x3_reduce', kernel_regularizer=l2(0.0002))(inception_4b_output)    
  inception_4c_3x3 = Conv2D(256, (3,3), padding='same',activation='relu',name='inception_4c_3x3', kernel_regularizer=l2(0.0002))(inception_4c_3x3_reduce)    
  inception_4c_5x5_reduce = Conv2D(24, (1,1), padding='same',activation='relu',name='inception_4c_5x5_reduce', kernel_regularizer=l2(0.0002))(inception_4b_output)    
  inception_4c_5x5 = Conv2D(64, (5,5), padding='same',activation='relu',name='inception_4c_5x5', kernel_regularizer=l2(0.0002))(inception_4c_5x5_reduce)    
  inception_4c_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='inception_4c_pool')(inception_4b_output)    
  inception_4c_pool_proj = Conv2D(64, (1,1), padding='same',activation='relu',name='inception_4c_pool_proj', kernel_regularizer=l2(0.0002))(inception_4c_pool)    
  inception_4c_output = Concatenate(axis=-1,name='inception_4c_output')([inception_4c_1x1,inception_4c_3x3,inception_4c_5x5,inception_4c_pool_proj])
  
  fintuning_extract_input_shape = inception_4c_output._keras_shape
  fintuning_extract_input = Input(shape=fintuning_extract_input_shape[1:], name='finetuning_extract_layer')

  inception_4d_1x1 = Conv2D(112, (1,1), padding='same',activation='relu',name='inception_4d_1x1', kernel_regularizer=l2(0.0002))(fintuning_extract_input)    
  inception_4d_3x3_reduce = Conv2D(144, (1,1), padding='same',activation='relu',name='inception_4d_3x3_reduce', kernel_regularizer=l2(0.0002))(fintuning_extract_input)    
  inception_4d_3x3 = Conv2D(288, (3,3), padding='same',activation='relu',name='inception_4d_3x3', kernel_regularizer=l2(0.0002))(inception_4d_3x3_reduce)    
  inception_4d_5x5_reduce = Conv2D(32, (1,1), padding='same',activation='relu',name='inception_4d_5x5_reduce', kernel_regularizer=l2(0.0002))(fintuning_extract_input)    
  inception_4d_5x5 = Conv2D(64, (5,5), padding='same',activation='relu',name='inception_4d_5x5', kernel_regularizer=l2(0.0002))(inception_4d_5x5_reduce)    
  inception_4d_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='inception_4d_pool')(fintuning_extract_input)    
  inception_4d_pool_proj = Conv2D(64, (1,1), padding='same',activation='relu',name='inception_4d_pool_proj', kernel_regularizer=l2(0.0002))(inception_4d_pool)    
  inception_4d_output = Concatenate(axis=-1,name='inception_4d_output')([inception_4d_1x1,inception_4d_3x3,inception_4d_5x5,inception_4d_pool_proj])
  
  inception_4e_1x1 = Conv2D(256, (1,1), padding='same',activation='relu',name='inception_4e_1x1', kernel_regularizer=l2(0.0002), )(inception_4d_output)    
  inception_4e_3x3_reduce = Conv2D(160, (1,1), padding='same',activation='relu',name='inception_4e_3x3_reduce', kernel_regularizer=l2(0.0002))(inception_4d_output)    
  inception_4e_3x3 = Conv2D(320, (3,3), padding='same',activation='relu',name='inception_4e_3x3', kernel_regularizer=l2(0.0002))(inception_4e_3x3_reduce)    
  inception_4e_5x5_reduce = Conv2D(32, (1,1), padding='same',activation='relu',name='inception_4e_5x5_reduce', kernel_regularizer=l2(0.0002))(inception_4d_output)   
  inception_4e_5x5 = Conv2D(128, (5,5), padding='same',activation='relu',name='inception_4e_5x5', kernel_regularizer=l2(0.0002))(inception_4e_5x5_reduce)    
  inception_4e_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='inception_4e_pool')(inception_4d_output)    
  inception_4e_pool_proj = Conv2D(128, (1,1), padding='same',activation='relu',name='inception_4e_pool_proj', kernel_regularizer=l2(0.0002))(inception_4e_pool)    
  inception_4e_output = Concatenate(axis=-1,name='inception_4e_output')([inception_4e_1x1,inception_4e_3x3,inception_4e_5x5,inception_4e_pool_proj])
  pool4_3x3_s2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same',name='pool4_3x3_s2')(inception_4e_output)

  inception_5a_1x1 = Conv2D(256, (1,1), padding='same',activation='relu',name='inception_5a_1x1', kernel_regularizer=l2(0.0002))(pool4_3x3_s2)    
  inception_5a_3x3_reduce = Conv2D(160, (1,1), padding='same',activation='relu',name='inception_5a_3x3_reduce', kernel_regularizer=l2(0.0002))(pool4_3x3_s2)    
  inception_5a_3x3 = Conv2D(320, (3,3), padding='same',activation='relu',name='inception_5a_3x3', kernel_regularizer=l2(0.0002))(inception_5a_3x3_reduce)    
  inception_5a_5x5_reduce = Conv2D(32, (1,1), padding='same',activation='relu',name='inception_5a_5x5_reduce', kernel_regularizer=l2(0.0002))(pool4_3x3_s2)    
  inception_5a_5x5 = Conv2D(128, (5,5), padding='same',activation='relu',name='inception_5a_5x5', kernel_regularizer=l2(0.0002))(inception_5a_5x5_reduce)    
  inception_5a_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='inception_5a_pool')(pool4_3x3_s2)    
  inception_5a_pool_proj = Conv2D(128, (1,1), padding='same',activation='relu',name='inception_5a_pool_proj', kernel_regularizer=l2(0.0002))(inception_5a_pool)    
  inception_5a_output = Concatenate(axis=-1,name='inception_5a_output')([inception_5a_1x1,inception_5a_3x3,inception_5a_5x5,inception_5a_pool_proj])
  
  inception_5b_1x1 = Conv2D(384, (1,1), padding='same',activation='relu',name='inception_5b_1x1', kernel_regularizer=l2(0.0002))(inception_5a_output)
  inception_5b_3x3_reduce = Conv2D(192, (1,1), padding='same',activation='relu',name='inception_5b_3x3_reduce', kernel_regularizer=l2(0.0002))(inception_5a_output)
  inception_5b_3x3 = Conv2D(384, (3,3), padding='same',activation='relu',name='inception_5b_3x3', kernel_regularizer=l2(0.0002))(inception_5b_3x3_reduce)    
  inception_5b_5x5_reduce = Conv2D(48, (1,1), padding='same',activation='relu',name='inception_5b_5x5_reduce', kernel_regularizer=l2(0.0002))(inception_5a_output)    
  inception_5b_5x5 = Conv2D(128, (5,5), padding='same',activation='relu',name='inception_5b_5x5', kernel_regularizer=l2(0.0002))(inception_5b_5x5_reduce)    
  inception_5b_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='inception_5b_pool')(inception_5a_output)    
  inception_5b_pool_proj = Conv2D(128, (1,1), padding='same',activation='relu',name='inception_5b_pool_proj', kernel_regularizer=l2(0.0002))(inception_5b_pool)    
  inception_5b_output = Concatenate(axis=-1,name='inception_5b_output')([inception_5b_1x1,inception_5b_3x3,inception_5b_5x5,inception_5b_pool_proj])
  
  finetuning_output = AveragePooling2D(pool_size=(7, 7), strides=(1,1), padding='valid')(inception_5b_output)
  output = Flatten()(finetuning_output)

  googlenet_base = Model(inputs=input, outputs=inception_4c_output)
  googlenet_finetuning = Model(inputs=fintuning_extract_input, outputs=output)
  
  weights = np.load(weights_path)

  def load_weights(model, weights):
    gen = (l for l in model.layers if l.name in weights.item().keys())
    for l in gen:
      ws = weights.item()[l.name]
      l.set_weights([ws['weights'], ws['biases']])

  load_weights(googlenet_base, weights)
  load_weights(googlenet_finetuning, weights)

  googlenet_base.save('{}_base.h5'.format(model_output_path))
  googlenet_finetuning.save('{}_finetuning.h5'.format(model_output_path))

  print('GoogleNet base model:')
  googlenet_base.summary()

  print('GoogleNet finetuning model:')
  googlenet_finetuning.summary()

  print('Done!')

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('-w', '--weights', required=True, 
    help='Path a NumPy array with weights (params) of GoogLeNet')
  parser.add_argument('-m', '--model-output', required=True,
    help='Path to an output model file')
  args = parser.parse_args()

  create_googlenet(args.weights, args.model_output)

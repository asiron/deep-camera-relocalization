from __future__ import print_function

from tqdm import tqdm

import numpy as np
import argparse, os, itertools

from models import pose_model, losses, metrics, layers
from utils import generate_images_from_filenames, make_dir
import cnn

import keras.backend as K

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from keras.models import Model
from keras.layers import (Input, Conv2D, 
  MaxPooling2D, GlobalAveragePooling2D, 
  Reshape, Dense, Dropout, Lambda, 
  CuDNNLSTM, Concatenate)

from vis.visualization import get_num_filters
from vis.utils import utils as utilsvis
from vis.visualization import (
  visualize_saliency, 
  visualize_activation,
  overlay)

from scipy.misc import imsave


def visualize_conv_filters(output_dir, model, layer_name):
  layer_idx = utilsvis.find_layer_idx(model, layer_name)

  filters = np.arange(get_num_filters(model.layers[layer_idx]))
  vis_images = []

  for idx in tqdm(filters):
      img = visualize_activation(model, layer_idx, filter_indices=idx)
      img = utilsvis.draw_text(img, 'Filter {}'.format(idx))    
      vis_images.append(img)

  stitched = utilsvis.stitch_images(vis_images, cols=32)    

  path = os.path.join(output_dir, '{}.png')
  imsave(path, stitched)


def get_smoothed_mask(func, input_seed, stdev_spread=.2, n_samples=50):
  """
  Returns a mask that is smoothed with the SmoothGrad method.

  Args:
      input_seed: input seed with shape (H, W, 3).
  """
  stdev = stdev_spread * (np.max(input_seed) - np.min(input_seed))

  gradient_shape = input_seed.shape[-3:-1]
  total_gradients = np.zeros(gradient_shape).astype(np.float)

  for _ in tqdm(xrange(n_samples), total=n_samples):
    noise = np.random.normal(0, stdev, input_seed.shape)
    total_gradients += func(input_seed + noise)

  return total_gradients / n_samples

def build_model(base_weights, top_weights):

  input = Input(shape=(224, 224, 3), name='cnn_input')
  
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

  conv5_1 = Conv2D(512, (3,3), strides=(1,1), activation='relu', name='conv5_1', padding='same')(pool4)
  conv5_2 = Conv2D(512, (3,3), strides=(1,1), activation='relu', name='conv5_2', padding='same')(conv5_1)
  conv5_3 = Conv2D(512, (3,3), strides=(1,1), activation='relu', name='conv5_3', padding='same')(conv5_2)
  pool5   = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='pool5', padding='same')(conv5_3)
  
  gap2d = GlobalAveragePooling2D()(pool5)

  dense_1 = Dense(1024, activation='relu')(gap2d)

  dropout_1 = Dropout(0.0001)(dense_1)

  reshaped = Reshape((32, 32))(dropout_1)
  reshaped_reversed = Lambda(lambda x: K.reverse(x, axes=1))(reshaped)

  transposed = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(reshaped)
  transposed_reversed = Lambda(lambda x: K.reverse(x, axes=1))(transposed)

  lstm_left_right = CuDNNLSTM(32, return_sequences=False)(reshaped)
  lstm_right_left = CuDNNLSTM(32, return_sequences=False)(reshaped_reversed)

  lstm_top_down = CuDNNLSTM(32, return_sequences=False)(transposed)
  lstm_bottom_up = CuDNNLSTM(32, return_sequences=False)(transposed_reversed)

  merged = Concatenate()([
    lstm_left_right, lstm_right_left, lstm_top_down, lstm_bottom_up])

  dense_2   = Dense(7)(merged)
  quat_norm = layers.QuaternionNormalization(name='quat_norm')(dense_2)
  prediction = Lambda(lambda x: x, name='prediction')(quat_norm)

  model = Model(inputs=input, outputs=prediction)

  model.load_weights(top_weights, by_name=True)
  model.load_weights(base_weights, by_name=True)

  tqdm.write('Model built!')

  return model

def main():
  parser = argparse.ArgumentParser()
  
  parser.add_argument('-i', '--images', nargs='+', required=True, 
    help='Paths to images to be visualzied with saliency')

  parser.add_argument('-o', '--output', required=True, 
    help='Path to an output dir where the saliency maps will be saved')
 
  parser.add_argument('-m', '--mode', default='vanilla',
    choices=['vanilla', 'guided'],
    help='saliency Map computation mode')
  
  parser.add_argument('--smooth', action='store_true', dest='smooth',
    help='Smooth the gradient')
  parser.add_argument('--no-smooth', action='store_false', dest='smooth',
    help='Do not smooth the gradient')
  parser.set_defaults(smooth=True)

  parser.add_argument('--model-weights',
    help='Model\'s weights to be used for saliency Map computation')

  parser.add_argument('-bs', '--batch-size', type=int, default=32,
    help='Batch size')

  args = parser.parse_args()

  base_weights = 'pose_regression/cnn/vgg16/hybrid1365/hybrid1365_base.h5'
  top_weights = args.model_weights
  

  custom_objects = {
    'QuaternionNormalization' : layers.QuaternionNormalization
  }

  images, _ = generate_images_from_filenames(args.images, batch_size=1)
  images = itertools.takewhile(lambda b: b is not None, images)

  '''
  model = build_model(base_weights, top_weights)

  conv_filters_output_dir = os.path.join(args.output, 'conv_filters')
  make_dir(conv_filters_output_dir)

  visualize_conv_filters(conv_filters_output_dir, model, 'conv5_3')

  grad_mods = ['negate', 'absolute', 'invert', 'relu', 'small_values']
  backprop_mods = ['relu', 'guided', None]
  '''
  grad_mods = ['absolute']
  backprop_mods = [None, 'guided']

  for n, img in enumerate(images):
    tqdm.write('Processing image {:5d}'.format(n))
    output_dir = os.path.join(args.output, '{:05d}_image'.format(n))

    for grad_mod in grad_mods:
      tqdm.write('Using grad modifier {}'.format(grad_mod))

      for backprop_mod in backprop_mods:
        backprop_mod_str = 'vanilla' if backprop_mod is None else backprop_mod
        tqdm.write('Using backprop modifier {}'.format(backprop_mod_str))
        
        path = os.path.join(output_dir, grad_mod, backprop_mod_str)
        make_dir(path)

        filename = os.path.join(path, '{}.png')

        def saliency(img):
          K.clear_session()
          model = build_model(base_weights, top_weights)
          return visualize_saliency(model, -1, None, img, 
            backprop_modifier=backprop_mod,
            grad_modifier=grad_mod,
            custom_objects=custom_objects)
       

        tqdm.write('Computing non-SmoothGrad gradients')
        grads = saliency(img)
        grads_rgb = np.uint8(cm.jet(grads)[..., :3] * 255)

        tqdm.write('Computing SmoothGrad gradients')
        smoothed_grads = get_smoothed_mask(saliency, img, n_samples=50)
        smoothed_grads_rgb = np.uint8(cm.jet(smoothed_grads)[..., :3] * 255)

        overlaid_grads = overlay(grads_rgb, np.squeeze(img))
        overlaid_smooothed_grads = overlay(smoothed_grads_rgb, np.squeeze(img))

        imsave(filename.format('plain'), grads_rgb)
        imsave(filename.format('overlaid'), overlaid_grads)
        imsave(filename.format('smoothgrad-plain'), smoothed_grads_rgb)
        imsave(filename.format('smoothgrad-overlaid'), overlaid_smooothed_grads)
  

if __name__ == '__main__':
  main()



  # params = args.model_weights.split('/')

  # top_model_type = params[-4].split('_')[0]
  # model_loss = params[-4].split('_')[1]
  # finetuning_model_arch = params[-4].split('_')[2]
  # finetuning_model_dataset = params[-4].split('_')[3]
  # seq_len = int(params[-4].split('_')[4].split('=')[1])

  # hyperparams = dict(map(lambda x: tuple(x.split('=')), params[-3].split(',')))

  # tqdm.write(hyperparams)
  

  # top_model = load_model(args.model_weights, custom_objects=custom_objects)
  # top_model_reduced = Model(inputs=top_model.input, outputs=top_model.get_layer('prediction').output)
  # top_model_reduced.name = 'top_model'

  # base_model = cnn.vgg16.vgg16.VGG16(mode='base', dataset='hybrid1365').build()
  # base_model.name = 'cnn_extractor'

  # base_model.summary()
  # top_model_reduced.summary()

  # model = Sequential()
  # model.add(base_model)
  # model.add(top_model_reduced)

  # model.summary()

  # trained_model = pose_model.PoseModel(
  #   input_shape=None,
  #   top_model_type=top_model_type,
  #   model_loss=model_loss,
  #   mode='predict',
  #   finetuning_model_arch=finetuning_model_arch,
  #   finetuning_model_dataset=finetuning_model_dataset,
  #   topmodel_weights=args.model_weights,
  #   seq_len=seq_len,
  #   **hyperparams).build()

  # trained_model.summary()
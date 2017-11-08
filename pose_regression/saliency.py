from __future__ import print_function

import numpy as np
import argparse, os, importlib, itertools

from keras.models import load_model, Sequential, Model
from keras.callbacks import (
  LearningRateScheduler, 
  ModelCheckpoint,
  EarlyStopping,
  ReduceLROnPlateau
)

from models import pose_model, losses, metrics, layers
from utils import (
  generate_images_from_filenames, make_dir, search_layer)
import cnn

import keras.backend as K

from vis.visualization import visualize_saliency, overlay
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from keras.layers import (Input, Conv2D, 
  MaxPooling2D, GlobalAveragePooling2D, 
  Reshape, Dense, Dropout, Lambda, 
  CuDNNLSTM, Concatenate)

import scipy.misc

def get_smoothed_mask(func, input_seed, stdev_spread=.2, n_samples=50):
  """
  Returns a mask that is smoothed with the SmoothGrad method.

  Args:
      input_seed: input seed with shape (H, W, 3).
  """
  stdev = stdev_spread * (np.max(input_seed) - np.min(input_seed))
  total_gradients = np.zeros((224, 224)).astype(np.float)

  for _ in xrange(n_samples):
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

  print('Built!')

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

  for n, img in enumerate(images):

    output_dir = os.path.join(args.output, '{:05d}_image'.format(n))
    make_dir(output_dir)

    for modifier in ['relu', 'guided', None]:


      figs = [plt.subplots() for _ in xrange(4)]

      titles = {
        'relu'   : 'ReLU',
        'guided' : 'Guided-Backpropgation',
            None : 'Vanilla'
      }
      map(lambda (f, ax): f.suptitle(titles[modifier]), figs)
      map(lambda (f, ax): ax.set_axis_off(), figs)

      filename = os.path.join(output_dir, '{}-{{}}.pdf'.format(modifier))

      def saliency(img):
        K.clear_session()
        model = build_model(base_weights, top_weights)
        return visualize_saliency(model, -1, None, img, 
                                            backprop_modifier=modifier,
                                            grad_modifier='absolute',
                                            custom_objects=custom_objects)

      grads = saliency(img)
      smoothed_grads = get_smoothed_mask(saliency, img, n_samples=5)

      f1, ax1 = figs[0]
      ax1.imshow(grads, cmap='jet')
      f1.suptitle(titles[modifier])
      f1.savefig(filename.format('plain'), bbox_inches='tight')

      f2, ax2 = figs[1]
      grad_rgb = np.uint8(cm.jet(grads)[..., :3] * 255)
      ax2.imshow(overlay(grad_rgb, np.squeeze(img)))

      f2.savefig(filename.format('overlayed'), bbox_inches='tight')

      f3, ax3 = figs[2]
      ax3.imshow(smoothed_grads, cmap='jet')
      f3.savefig(filename.format('smoothgrad-plain'), bbox_inches='tight')

      f4, ax4 = figs[3]
      grad_rgb = np.uint8(cm.jet(smoothed_grads)[..., :3] * 255)
      ax4.imshow(overlay(grad_rgb, np.squeeze(img)))
      f4.savefig(filename.format('smoothgrad-overlayed'), bbox_inches='tight')

if __name__ == '__main__':
  main()



  # params = args.model_weights.split('/')

  # top_model_type = params[-4].split('_')[0]
  # model_loss = params[-4].split('_')[1]
  # finetuning_model_arch = params[-4].split('_')[2]
  # finetuning_model_dataset = params[-4].split('_')[3]
  # seq_len = int(params[-4].split('_')[4].split('=')[1])

  # hyperparams = dict(map(lambda x: tuple(x.split('=')), params[-3].split(',')))

  # print(hyperparams)
  

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
import os
import numpy as np

from .googlenet_layers import LRN

from keras.layers import GlobalAveragePooling2D
from keras.models import Model, load_model, Sequential

class GoogleNet(object):

  MODES = ['extract', 'finetune']
  DATASETS = ['imagenet', 'places365']

  def __init__(self, dataset=None, mode=None):

    if mode not in GoogleNet.MODES:
      raise ValueError('Must specify a valid mode!')

    if dataset not in GoogleNet.DATASETS:
      raise ValueError('Must specify a valid dataset!')

    weights_dir = os.path.join(os.path.dirname(__file__), dataset)

    self.input_shape = (224, 224, 3)
    self.bgr_mean_file = np.load(os.path.join(weights_dir, 'meanfile.npy'))
    
    last_inception_block_file = os.path.join(
      weights_dir, '{}_last_inception.h5'.format(dataset))
    last_inception_block = load_model(last_inception_block_file)

    if mode == 'extract':

      googlenet_base_file = os.path.join(
        weights_dir, '{}_base.h5'.format(dataset))
      googlenet_base = load_model(googlenet_base_file, custom_objects={'LRN': LRN})

      model = Sequential()
      model.add(googlenet_base)
      model.add(last_inception_block)

      self.model = Model(inputs=googlenet_base.input,
        outputs=[model.output, googlenet_base.output])

    elif mode == 'finetune':
      self.model = last_inception_block

  def preprocess_image(self, images):
    '''
    Assumes that images are in RGB and converts to BGR, then subtracts the mean
    No scaling to [0,1] range is required as the models were trained using Caffe
    '''    
    images = images[::-1, ...]
    images[..., 0] -= self.bgr_mean_file[0]
    images[..., 1] -= self.bgr_mean_file[1]
    images[..., 2] -= self.bgr_mean_file[2]
    return images


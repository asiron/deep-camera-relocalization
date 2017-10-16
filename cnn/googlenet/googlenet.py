import os
import numpy as np

from .googlenet_layers import LRN

from keras.layers import Input
from keras.models import Model, load_model, Model

class GoogleNet(object):

  MODES = ['extract', 'finetune']
  DATASETS = ['imagenet', 'places365', 'places205']

  def __init__(self, dataset=None, mode=None, meanfile=None):

    if mode not in GoogleNet.MODES:
      raise ValueError('Must specify a valid mode!')

    if dataset not in GoogleNet.DATASETS:
      raise ValueError('Must specify a valid dataset!')

    if mode == 'extract' and not meanfile:
      raise ValueError('Must specify a valid meanfile!')

    if meanfile:
      self.rgb_mean_file = np.load(meanfile)

    weights_dir = os.path.join(os.path.dirname(__file__), dataset)

    self.input_shape = (224, 224, 3)
    
    last_inception_block_file = os.path.join(
      weights_dir, '{}_last_inception.h5'.format(dataset))
    last_inception_block = load_model(last_inception_block_file)

    if mode == 'extract':

      googlenet_base_file = os.path.join(
        weights_dir, '{}_base.h5'.format(dataset))
      googlenet_base = load_model(googlenet_base_file, custom_objects={'LRN': LRN})

      input_layer = Input(shape=self.input_shape, name='extract_input')

      base = googlenet_base(input_layer)
      heads = last_inception_block(base)

      if isinstance(heads, list):
        outputs = [heads[0], heads[1], heads[2], base]
      else:
        outputs = [heads, base]
        
      self.model = Model(inputs=input_layer, outputs=outputs)

    elif mode == 'finetune':
      self.model = last_inception_block

  def preprocess_image(self, image):
    '''
    Subtracts the mean and converts from RGB to BGR
    No scaling to [0,1] range is required as the models were trained using Caffe
    '''    
    image -= self.rgb_mean_file
    image = image[..., ::-1]
    return image


from __future__ import print_function

import os
import numpy as np

from keras.layers import Input
from keras.models import Model, load_model, Model

from .googlenet_layers import LRN

class GoogleNet(object):

  MODES = ['extract', 'finetune']
  DATASETS = ['imagenet', 'places365', 'places205']

  def __init__(self, mode=None, **kwargs):

    if mode not in GoogleNet.MODES:
      raise ValueError('Must specify a valid mode!')

    if kwargs.get('dataset', None) not in GoogleNet.DATASETS:
      raise ValueError('Must specify a valid dataset!')

    if mode == 'extract' and kwargs.get('meanfile', None) is None:
      raise ValueError('Must specify a valid meanfile!')

    meanfile = kwargs.get('meanfile', None)
    if meanfile is not None:
      print('Meanfile loaded from {}'.format(meanfile))
      self.rgb_channelwise_mean = np.load(meanfile).mean(axis=(0,1))

    self.mode = mode
    self.dataset = kwargs['dataset']
    self.input_shape = (224, 224, 3)

  def build(self):

    weights_dir = os.path.join(os.path.dirname(__file__), self.dataset)

    finetuning_model_file = os.path.join(
      weights_dir, '{}_finetuning.h5'.format(self.dataset))
    finetuning_model = load_model(finetuning_model_file)

    googlenet_base_file = os.path.join(
      weights_dir, '{}_base.h5'.format(self.dataset))
    googlenet_base = load_model(googlenet_base_file, custom_objects={'LRN': LRN})

    if self.mode == 'extract':
      input_layer = Input(shape=self.input_shape, name='extract_input')
      base = googlenet_base(input_layer)
      heads = finetuning_model(base)
      return Model(inputs=input_layer, outputs=[heads, base])

    elif self.mode == 'finetune':
      return finetuning_model

    elif self.mode == 'base':
      return googlenet_base


  def preprocess_image(self, image):
    '''
    Subtracts the channel-wise mean and converts from RGB to BGR
    No scaling to [0,1] range is required as the models were trained using Caffe
    '''    
    image[..., 0] -= self.rgb_channelwise_mean[0]
    image[..., 1] -= self.rgb_channelwise_mean[1]
    image[..., 2] -= self.rgb_channelwise_mean[2]
    image = image[..., ::-1]
    return image


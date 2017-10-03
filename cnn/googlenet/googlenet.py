import os

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
      model.add(GlobalAveragePooling2D())

      self.model = Model(inputs=googlenet_base.input,
        outputs=[googlenet_base.output, model.output])

    elif mode == 'finetune':

      self.model = last_inception_block

  def preprocess_image(self, images):
    '''
    Assumes that images are in RGB and converts to BGR
    '''
    images[:,:,[0,1,2]] = images[:,:,[2,1,0]]
    return images


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

    self.input_shape = (224, 224, 3)

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

      self.model = Model(inputs=googlenet_base.input,
        outputs=[model.output, googlenet_base.output])

    elif mode == 'finetune':
      self.model = last_inception_block

  @staticmethod
  def preprocess_image(img):
    '''
    Assumes that img are in RGB and converts to BGR
    '''
    img[...,[0,1,2]] = img[...,[2,1,0]]
    return img


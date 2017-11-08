from keras.layers import Input, GlobalAveragePooling2D, AveragePooling2D
from keras.models import Model, Sequential

from .inception_resnet_v2_base import preprocess_input, inception_resnet_block, conv2d_bn, load_weights
from .inception_resnet_v2_base import InceptionResNetV2 as BaseModel

from ..image_utils.image_utils import scale_image

class InceptionResNetV2(object):

  MODES = ['extract', 'finetune']
  DATASETS = ['imagenet']

  def __init__(self, mode=None, **kwargs):

    if mode not in InceptionResNetV2.MODES:
      raise ValueError('Must specify a valid mode!')

    if kwargs.get('dataset', None) not in InceptionResNetV2.DATASETS:
      raise ValueError('Must specify a valid dataset!')

    self.input_shape = (299, 299, 3)
    self.dataset = kwargs['dataset']
    self.mode = mode

  def build(self):

    base_model = BaseModel(
      weights=self.dataset, 
      include_top=False, 
      input_shape=self.input_shape,
      pooling='avg')

    injection_layer_name = 'before_last_conv_passthrough'
    injection_layer = base_model.get_layer(injection_layer_name)
    '''idx = [i for i,x in enumerate(base_model.layers) if x.name == injection_layer_name][0]'''

    if self.mode == 'extract':
      self.model = Model(inputs=base_model.input,
        outputs=[base_model.output, injection_layer.output])

    elif self.mode == 'finetune':
      top_model_input = Input(
        shape=injection_layer.input_shape[1:], 
        name='last_conv_input')
      x = top_model_input

      # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
      for block_idx in range(1, 10):
          x = inception_resnet_block(x,
                                     scale=0.2,
                                     block_type='block8',
                                     block_idx=block_idx)
      x = inception_resnet_block(x,
                                 scale=1.,
                                 activation=None,
                                 block_type='block8',
                                 block_idx=10)

      # Final convolution block: 8 x 8 x 1536
      x = conv2d_bn(x, 1536, 1, name='conv_7b')
      finetune_model_output = GlobalAveragePooling2D()(x)

      self.model = Model(inputs=top_model_input, outputs=finetune_model_output)
      self.model = load_weights(self.model, include_top=False, weights=self.dataset, by_name=True)

    return self.model

  def preprocess_image(self, images):
    '''Inception ResNet v2 was traing on RGB images only scaling to [-1, 1] is required'''
    return scale_image(images)

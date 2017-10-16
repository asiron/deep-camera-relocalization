from keras.layers import Input, GlobalAveragePooling2D
from keras.models import Model, Sequential

from .inception_resnet_v2_base import preprocess_input, inception_resnet_block, conv2d_bn
from .inception_resnet_v2_base import InceptionResNetV2 as BaseModel

from ..image_utils.image_utils import scale_image
class InceptionResNetV2(object):

  MODES = ['extract', 'finetune']
  DATASETS = ['imagenet']

  def __init__(self, dataset='imagenet', mode=None):

    if mode not in InceptionResNetV2.MODES:
      raise ValueError('Must specify a valid mode!')

    if dataset not in InceptionResNetV2.DATASETS:
      raise ValueError('Must specify a valid dataset!')

    self.input_shape = (299, 299, 3)

    base_model = BaseModel(weights=dataset, 
      include_top=False, 
      input_shape=self.input_shape,
      pooling='avg')

    injection_layer_name = 'before_last_conv_passthrough'
    idx = [i for i,x in enumerate(base_model.layers) if x.name == injection_layer_name][0]
    injection_layer = base_model.layers[idx]

    if mode == 'extract':

      self.model = Model(
        inputs=base_model.input,
        outputs=[base_model.output, injection_layer.output])
    
    elif mode == 'finetune':

      top_model_input = Input(shape=injection_layer.input_shape[1:], name='last_conv_input')
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
      top_model_output = GlobalAveragePooling2D()(x)
      #top_model_output = reduce(lambda x,l: l(x), base_model.layers[idx:], top_model_input)



      top_model = Model(inputs=top_model_input, outputs=top_model_output)
      self.model = top_model

  def preprocess_image(self, images):
    '''Inception ResNet v2 was traing on RGB images so no need to convert'''
    return scale_image(images)

from keras.layers import Input
from keras.models import Model, Sequential

from .inception_resnet_v2_base import InceptionResNetV2 as BaseModel

class InceptionResNetV2(object):

  MODES = ['extract', 'finetune']
  DATASETS = ['imagenet']

  def __init__(self, dataset='imagenet', mode=None):

    if mode not in InceptionResnetV2.MODES:
      raise ValueError('Must specify a valid mode!')

    if dataset not in InceptionResnetV2.DATASETS:
      raise ValueError('Must specify a valid dataset!')

    self.input_shape = (299, 299, 3)

    base_model = BaseModel(weights=dataset, 
      include_top=False, 
      input_shape=self.input_shape,
      pooling='avg')

    injection_layer_name = 'before_last_conv_passthrough'
    idx = [i for i,x in enumerate(a.layers) if x.name == injection_layer_name][0]

    injection_layer = base_model.layers[idx]
    top_model_input = Input(shape=injection_layer.input_shape[1:], name='last_conv_input')
    top_model_output = reduce(lambda x,l: l(x), base_model.layers[idx:], x)
    top_model = Model(inputs=top_model_input, outputs=top_model_output)

    if mode == 'extract':

      inception_base = Model(inputs=base_model.input, outputs=injection_layer.input)
      
      model = Sequential()
      model.add(inception_base)
      model.add(top_model)
      
      self.model = Model(inputs=inception_base.input,
        outputs=[inception_base.output, model.output])

    elif mode == 'finetune':

      self.model = top_model

  def preprocess_image(self, images):
    '''
    Inception ResNet v2 was traing on RGB images
    so no need to convert
    '''
    return images

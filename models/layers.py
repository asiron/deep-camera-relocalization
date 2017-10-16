from keras.layers import Lambda
from keras.engine.topology import Layer
from keras.initializers import Constant
import keras.backend as K

class QuaternionNormalization(Lambda):

  def __init__(self, name=None):
    
    def layer(x):
      pos, quat = x[..., :3], x[..., 3:]
      quat = K.l2_normalize(quat, axis=-1)
      return K.concatenate([pos, quat], axis=-1)
    
    super(QuaternionNormalization, self).__init__(layer, name=name)

class HomoscedasticLoss(Layer):

  def __init__(self, log_variance_init=0.0, **kwargs):
    super(HomoscedasticLoss, self).__init__(**kwargs)
    self.log_variance_init = log_variance_init

  def build(self, input_shape):

    log_variance_initializer = Constant(value=self.log_variance_init)
    self.log_variance = self.add_weight(name='log_variance', 
                                        shape=(1,),
                                        initializer=log_variance_initializer,
                                        trainable=True)
    super(HomoscedasticLoss, self).build(input_shape)

  def call(self, x):
    return x * K.exp(-self.log_variance) + self.log_variance

  def compute_output_shape(self, input_shape):
    return (input_shape[0], 1)

  def get_config(self):
    config = {'log_variance_init': self.log_variance_init}
    base_config = super(HomoscedasticLoss, self).get_config()
    print base_config
    return dict(list(base_config.items()) + list(config.items()))

from keras.layers import Lambda
import keras.backend as K

class QuaternionNormalization(Lambda):

  def __init__(self, name=None):
    
    def layer(x):
      pos, quat = x[..., :3], x[..., 3:]
      quat = K.l2_normalize(quat, axis=-1)
      return K.concatenate([pos, quat], axis=-1)
    
    super(QuaternionNormalization, self).__init__(layer, name=name)

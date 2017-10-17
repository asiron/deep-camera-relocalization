from keras.layers import Layer

import tensorflow as tf

class LRN(Layer):

  def __init__(self, radius=2, alpha=2e-5, beta=0.75, bias=1.0, **kwargs):
    self.radius = radius
    self.alpha = alpha
    self.beta = beta
    self.bias = bias
    super(LRN, self).__init__(**kwargs)
  
  def call(self, x, mask=None):
    return tf.nn.local_response_normalization(x,
      depth_radius=self.radius,
      alpha=self.alpha,
      beta=self.beta,
      bias=self.bias,
      name=self.name)

  def get_config(self):
    config = {
      "radius": self.radius,
      "alpha": self.alpha,
      "beta": self.beta,
      "bias": self.bias}
    base_config = super(LRN, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

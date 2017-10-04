
from keras.models import Model
from keras.layers import Dropout, Dense, Input
from keras.optimizers import Adam
from keras.regularizers import l2

from .layers import QuaternionNormalization

class Regressor(object):

  def __init__(self, input_shape, **kwargs):

    self.kwargs = kwargs

    inputs = Input(shape=input_shape)

    x = Dense(input_shape[0],
      activation='relu', 
      W_regularizer=l2(self.kwargs['l2_regu']))(inputs)
    x = Dropout(self.kwargs['dropout'])(x)
    x = Dense(7)(x)
    x = QuaternionNormalization(name='quat_norm')(x)

    self.model = Model(inputs=inputs, outputs=x)



  #   self.optimizer = Adam(lr=self.kwargs['l_rate'])

  # def build(self):
  #   raise NotImplementedError('build method must be implemented in subclass!')

# class NaiveWeightedLinearRegression(WeightedLinearRegression):

#   def build(self):
#     loss = NaiveWeightedPoseLoss(
#       beta=self.kwargs['beta'],
#       gamma=self.kwargs['gamma'])
#     self.model.compile(optimizer=self.optimizer, loss=loss)
#     self.model.summary()
#     return self.model

# class ProperWeightedLinearRegression(WeightedLinearRegression):

#   def build(self):
#     loss = ProperWeightedPoseLoss(
#       beta=self.kwargs['beta'],
#       gamma=self.kwargs['gamma'])
#     self.model.compile(optimizer=self.optimizer, loss=loss)
#     return self.model


from keras.models import Model
from keras.layers import Dropout, Dense, Input
from keras.optimizers import Adam
from keras.regularizers import l2

from .layers import QuaternionNormalization

class Regressor(object):

  #def __init__(self, input_shape, **kwargs):
  def __init__(self, **kwargs):

    self.kwargs = kwargs
    #self.input = Input(shape=input_shape)

  def build(self, input_tensor):


    dense_1 = Dense(int(input_tensor.shape[1]),
      activation='relu', 
      W_regularizer=l2(self.kwargs['l2_regu']))(input_tensor)
    dropout_1 = Dropout(self.kwargs['dropout'])(dense_1)
    dense_2 = Dense(7)(dropout_1)
    quat_norm = QuaternionNormalization(name='quat_norm')(dense_2)
    return quat_norm
    #self.model = Model(inputs=inputs, outputs=quat_norm)

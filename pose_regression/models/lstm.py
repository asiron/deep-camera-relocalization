from keras.models import Model
from keras.layers import Dropout, Dense, Input, LSTM, TimeDistributed
from keras.optimizers import Adam
from keras.regularizers import l2

from .layers import QuaternionNormalization

class LSTM(object):

  def __init__(self, **kwargs):

    self.kwargs = kwargs
    #self.input = Input(shape=input_shape)

  def build(self, input_tensor):

    input_shape = int(input_tensor.shape[1])

    lstm1 = LSTM(input_shape, return_sequences=True)(input_tensor)
    #lstm2 = LSTM(int(input_tensor.shape[1]), return_sequences=True)(lstm1)
    #lstm2 = LSTM(int(input_tensor.shape[1]), return_sequences=True)(lstm1)
    dense_1   = TimeDistributed(Dense(input_shape, activation='relu'))(lstm1)
    dropout_1 = TimeDistributed(Dropout(self.kwargs['dropout']))(dense_1)
    dense_2   = TimeDistributed(Dense(7))(dropout_1)
    quat_norm = TimeDistributed(QuaternionNormalization(name='quat_norm'))(dense_2)
    return quat_norm

from keras.layers import Dropout, Dense, LSTM, GRU, CuDNNLSTM, Lambda, TimeDistributed, Bidirectional, Reshape, Concatenate
from keras.regularizers import l2

from .layers import QuaternionNormalization

import keras.backend as K

import tensorflow as tf

class TopModel(object):

  def __init__(self, **kwargs):
    self.kwargs = kwargs

  def build(self, input_tensor):
    return Lambda(lambda x: x, name='prediction')(input_tensor)

class Regressor(TopModel):

  def build(self, input_tensor):
    
    assert len(input_tensor._keras_shape[1:]) in [1,2]
    
    dense_1   = Dense(input_tensor._keras_shape[1],
                      activation='relu', 
                      W_regularizer=l2(self.kwargs['l2_regu']))(input_tensor)
    dropout_1 = Dropout(self.kwargs['dropout'])(dense_1)
    dense_2   = Dense(7)(dropout_1)
    quat_norm = QuaternionNormalization(name='quat_norm')(dense_2)
    return super(Regressor, self).build(quat_norm)


class SpatialLSTM(TopModel):

  def build(self, input_tensor):
    
    assert len(input_tensor._keras_shape[1:]) is 1
    
    dense_1   = Dense(1024,
                      activation='relu', 
                      W_regularizer=l2(self.kwargs['l2_regu']))(input_tensor)

    reshaped = Reshape((32, 32))(dense_1)
    reshaped_reversed = Lambda(lambda x: K.reverse(x, axes=1))(reshaped)

    transposed = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(reshaped)
    transposed_reversed = Lambda(lambda x: K.reverse(x, axes=1))(transposed)

    print(reshaped.shape, transposed.shape)

    lstm_left_right = CuDNNLSTM(32, return_sequences=False)(reshaped)
    lstm_right_left = CuDNNLSTM(32, return_sequences=False)(reshaped_reversed)

    lstm_top_down = CuDNNLSTM(32, return_sequences=False)(transposed)
    lstm_bottom_up = CuDNNLSTM(32, return_sequences=False)(transposed_reversed)


    # lstm_left_right = LSTM(32, return_sequences=False)(reshaped)
    # lstm_right_left = LSTM(32, return_sequences=False, go_backwards=True)(reshaped)

    # lstm_top_down = LSTM(32, return_sequences=False)(transposed)
    # lstm_bottom_up = LSTM(32, return_sequences=False, go_backwards=True)(transposed)

    merged = Concatenate()([
      lstm_left_right, lstm_right_left, lstm_top_down, lstm_bottom_up])

    dropout_1 = Dropout(self.kwargs['dropout'])(merged)

    dense_2   = Dense(7)(dropout_1)
    quat_norm = QuaternionNormalization(name='quat_norm')(dense_2)
    return super(SpatialLSTM, self).build(quat_norm)


class StatefulLSTM(TopModel):

  def build(self, input_tensor):

    lstm_units = self.kwargs['lstm_units']
    lstm1 = CuDNNLSTM(lstm_units, return_sequences=False, stateful=True)(input_tensor)
    #lstm2 = CuDNNLSTM(lstm_units//2, return_sequences=False, stateful=True)(lstm1)

    assert len(input_tensor.shape[2:]) == 1

    dense_1   = Dense(lstm_units//2,
                      activation='relu', 
                      W_regularizer=l2(self.kwargs['l2_regu']))(lstm2)
    dropout_1 = Dropout(self.kwargs['dropout'])(lstm1)
    dense_2   = Dense(7)(dropout_1)
    
    quat_norm = QuaternionNormalization(name='quat_norm')(dense_2)
    return super(StatefulLSTM, self).build(quat_norm)

class StatelessLSTM(TopModel):

  def build(self, input_tensor):

    assert len(input_tensor.shape[2:]) == 1

    # dense_1   = TimeDistributed(Dense(input_tensor._keras_shape[2],
    #                   activation='relu', 
    #                   W_regularizer=l2(self.kwargs['l2_regu'])))(input_tensor)
    # dropout_1 = TimeDistributed(Dropout(self.kwargs['dropout']))(dense_1)
    # dense_2   = TimeDistributed(Dense(7))(dropout_1)
    # quat_norm = TimeDistributed(QuaternionNormalization(name='quat_norm'))(dense_2)

    # lstm_units = self.kwargs['lstm_units']
    # lstm1 = LSTM(lstm_units, return_sequences=True)(quat_norm)
    
    # dense_3   = TimeDistributed(Dense(7))(lstm1)
    # quat_norm_2 = TimeDistributed(QuaternionNormalization(name='quat_norm_2'))(dense_3)
    # return TimeDistributed(Lambda(lambda x: x, name='prediction'), name='prediction')(quat_norm_2)
    #
    dense_1   = TimeDistributed(Dense(input_tensor._keras_shape[2],
                                      activation='relu', 
                                      W_regularizer=l2(self.kwargs['l2_regu'])))(input_tensor)

    dropout_1 = TimeDistributed(Dropout(self.kwargs['dropout']))(dense_1)
    dense_2   = TimeDistributed(Dense(7))(dropout_1)

    lstm_units = self.kwargs['lstm_units']
    lstm1 = GRU(lstm_units, return_sequences=True)(dense_2)

    dense_3   = TimeDistributed(Dense(7))(lstm1)

    #dense_1 = TimeDistributed(Dense(lstm_units, activation='relu'))(lstm1)
    # dense_1 = TimeDistributed(Dense(lstm_units,
    #  activation='relu', kernel_regularizer=l2(self.kwargs['l2_regu'])))(lstm1)


    quat_norm = TimeDistributed(QuaternionNormalization(name='quat_norm'))(dense_3)
    return TimeDistributed(Lambda(lambda x: x, name='prediction'), name='prediction')(quat_norm)

    # lstm_units = self.kwargs['lstm_units']
    # lstm1 = Bidirectional(LSTM(lstm_units, return_sequences=True
    #   #recurrent_dropout=0.2, 
    #   #kernel_regularizer=l2(0.01),
    #   #bias_regularizer=l2(0.01),
    #   #recurrent_regularizer=l2(0.01)
    #   ), merge_mode='concat')(input_tensor)

    # lstm1_dense = TimeDistributed(Dense(lstm_units))(lstm1)

    # lstm2 = Bidirectional(LSTM(lstm_units, return_sequences=True
    #   #recurrent_dropout=0.2, 
    #   #kernel_regularizer=l2(0.01),
    #   #bias_regularizer=l2(0.01),
    #   #recurrent_regularizer=l2(0.01)
    #   ), merge_mode='concat')(lstm1_dense)

    # # dense_1 = TimeDistributed(Dense(lstm_units,
    # #  activation='relu', kernel_regularizer=l2(self.kwargs['l2_regu'])))(lstm1)

    # #dropout_1 = TimeDistributed(Dropout(self.kwargs['dropout']))(lstm2)
    # dense_2   = TimeDistributed(Dense(7))(lstm2)
    # quat_norm = TimeDistributed(QuaternionNormalization(name='quat_norm'))(dense_2)
    # return TimeDistributed(Lambda(lambda x: x, name='prediction'), name='prediction')(quat_norm)



class RegressorLSTM(TopModel):

  def build(self, input_tensor):
    
    reshaped_input_tensor = Lambda(lambda x: tf.squeeze(x))(input_tensor)

    dense_1   = Dense(int(reshaped_input_tensor.shape[1]),
                      activation='relu', 
                      W_regularizer=l2(self.kwargs['l2_regu']))(reshaped_input_tensor)
    dropout_1 = Dropout(self.kwargs['dropout'])(dense_1)

    dense_2 = Dense(7)(dropout_1)

    dense_2_reshaped = Lambda(lambda x: tf.expand_dims(x, axis=1))(dense_2)

    lstm_units = self.kwargs['lstm_units']
    lstm1 = LSTM(lstm_units, return_sequences=False, stateful=True)(dense_2_reshaped)

    dense_3 = Dense(7)(lstm1)
    quat_norm = QuaternionNormalization(name='quat_norm')(dense_3)

    return super(RegressorLSTM, self).build(quat_norm)

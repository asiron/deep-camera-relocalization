from keras.layers import (
    Dropout, Dense, LSTM, ELU, GRU, CuDNNGRU, CuDNNLSTM, Lambda, 
    TimeDistributed, Activation, Bidirectional, 
    Reshape, Concatenate, PReLU, BatchNormalization)

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
    
    # dense_1   = Dense(input_tensor._keras_shape[1],
    #                   activation='relu', 
    #                   ))(input_tensor)
                      
    dense_1   = Dense(self.kwargs['units'], 
                      activation='relu',
                      kernel_regularizer=l2(self.kwargs['l2']))(input_tensor)

    dropout_1 = Dropout(self.kwargs['dropout'])(dense_1)
    dense_2   = Dense(7)(dropout_1)
    quat_norm = QuaternionNormalization(name='quat_norm')(dense_2)
    return super(Regressor, self).build(quat_norm)


class SpatialLSTM(TopModel):

  def build(self, input_tensor):
    
    assert len(input_tensor._keras_shape[1:]) is 1
    
    dense_1 = Dense(2048,
                    activation='relu', 
                    kernel_regularizer=l2(self.kwargs['l2']))(input_tensor)

    rect_shape = (64, 32)

    dropout_1 = Dropout(self.kwargs['dropout'])(dense_1)

    reshaped = Reshape(rect_shape)(dropout_1)
    reshaped_reversed = Lambda(lambda x: K.reverse(x, axes=1))(reshaped)

    transposed = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(reshaped)
    transposed_reversed = Lambda(lambda x: K.reverse(x, axes=1))(transposed)

    lstm_top_down  = CuDNNLSTM(rect_shape[0], return_sequences=False)(reshaped)
    lstm_bottom_up = CuDNNLSTM(rect_shape[0], return_sequences=False)(reshaped_reversed)

    lstm_left_right = CuDNNLSTM(rect_shape[1], return_sequences=False)(transposed)
    lstm_right_left = CuDNNLSTM(rect_shape[1], return_sequences=False)(transposed_reversed)

    merged = Concatenate()([
      lstm_left_right, lstm_right_left, 
      lstm_top_down,   lstm_bottom_up
    ])

    dense_2   = Dense(7)(merged)
    quat_norm = QuaternionNormalization(name='quat_norm')(dense_2)
    return super(SpatialLSTM, self).build(quat_norm)


class StatefulLSTM(TopModel):

  def build(self, input_tensor):

    assert len(input_tensor.shape[2:]) == 1

    lstm_units = self.kwargs['units']

    #dense_1 = TimeDistributed(Dense(1024, activation='relu'))(input_tensor)
    lstm_1 = CuDNNGRU(512, return_sequences=True, stateful=True)(input_tensor)
    dense_2   = TimeDistributed(Dense(7))(lstm_1)
    quat_norm = TimeDistributed(QuaternionNormalization(name='quat_norm'))(dense_2)
    return TimeDistributed(Lambda(lambda x: x, name='inner_prediction'), name='prediction')(quat_norm)


class StandardLSTM(TopModel):

  def build(self, input_tensor):

    assert len(input_tensor.shape[2:]) == 1

    lstm_units = self.kwargs['units']

    dense_1   = TimeDistributed(Dense(1024))(input_tensor)

    merged_rev = TimeDistributed(Lambda(lambda x: K.reverse(x, axes=1)))(dense_1)

    lstm_1 = CuDNNLSTM(512, return_sequences=True)(dense_1)
    lstm_1_rev = CuDNNLSTM(512, return_sequences=True)(merged_rev)
    
    merged_bidirectional = Concatenate(axis=-1)([lstm_1, lstm_1_rev])

    dropout_1 = Dropout(self.kwargs['dropout'])(merged_bidirectional)

    dense_2   = TimeDistributed(Dense(7))(dropout_1)
    quat_norm = TimeDistributed(QuaternionNormalization(name='quat_norm'))(dense_2)
    return TimeDistributed(Lambda(lambda x: x, name='inner_prediction'), name='prediction')(quat_norm)


    '''
    dense_1 = TimeDistributed(Dense(2048,
                    activation='relu', 
                    kernel_regularizer=l2(self.kwargs['l2'])))(merged_bidirectional)

    rect_shape = (64, 32)

    reshaped = TimeDistributed(Reshape(rect_shape))(dense_1)
    reshaped_reversed = TimeDistributed(Lambda(lambda x: K.reverse(x, axes=1)))(reshaped)

    transposed = TimeDistributed(Lambda(lambda x: K.permute_dimensions(x, (0,2,1))))(reshaped)
    transposed_reversed = TimeDistributed(Lambda(lambda x: K.reverse(x, axes=1)))(transposed)

    lstm_top_down  = TimeDistributed(CuDNNLSTM(rect_shape[0], return_sequences=False))(reshaped)
    lstm_bottom_up = TimeDistributed(CuDNNLSTM(rect_shape[0], return_sequences=False))(reshaped_reversed)

    lstm_left_right = TimeDistributed(CuDNNLSTM(rect_shape[1], return_sequences=False))(transposed)
    lstm_right_left = TimeDistributed(CuDNNLSTM(rect_shape[1], return_sequences=False))(transposed_reversed)

    merged_spatial = Concatenate(axis=-1)([
      lstm_left_right, lstm_right_left, 
      lstm_top_down,   lstm_bottom_up
    ])

    merged_spatial_rev = TimeDistributed(Lambda(lambda x: K.reverse(x, axes=1)))(merged_spatial)

    lstm_2 = CuDNNLSTM(2*rect_shape[0] + 2*rect_shape[1], return_sequences=True)(merged_spatial)
    lstm_2_rev = CuDNNLSTM(512, return_sequences=True)(merged_spatial_rev)
    
    merged_bidirectional = Concatenate(axis=-1)([lstm_2, lstm_2_rev])

    dense_2   = TimeDistributed(Dense(7))(merged_bidirectional)
    quat_norm = TimeDistributed(QuaternionNormalization(name='quat_norm'))(dense_2)
    return TimeDistributed(Lambda(lambda x: x, name='inner_prediction'), name='prediction')(quat_norm)
    '''
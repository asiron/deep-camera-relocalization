from keras.layers import (
    Dropout, Dense, LSTM, GRU, CuDNNLSTM, Lambda, 
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
    #                   W_regularizer=l2(self.kwargs['l2_regu']))(input_tensor)
                      
    dense_1   = Dense(input_tensor._keras_shape[1],
                      activation='relu')(input_tensor)
    dropout_1 = Dropout(self.kwargs['dropout'])(dense_1)
    dense_2   = Dense(7)(dropout_1)
    quat_norm = QuaternionNormalization(name='quat_norm')(dense_2)
    return super(Regressor, self).build(quat_norm)


class SpatialLSTM(TopModel):

  def build(self, input_tensor):
    
    assert len(input_tensor._keras_shape[1:]) is 1
    
    # dense_1 = Dense(1024,
    #                 activation='relu', 
    #                 kernel_regularizer=l2(self.kwargs['l2_regu']))(input_tensor)

    dense_1 = Dense(1024, activation='relu')(input_tensor)

    dropout_1 = Dropout(self.kwargs['dropout'])(dense_1)

    reshaped = Reshape((32, 32))(dropout_1)
    reshaped_reversed = Lambda(lambda x: K.reverse(x, axes=1))(reshaped)

    transposed = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(reshaped)
    transposed_reversed = Lambda(lambda x: K.reverse(x, axes=1))(transposed)

    print(reshaped.shape, transposed.shape)

    lstm_left_right = CuDNNLSTM(32, return_sequences=False)(reshaped)
    lstm_right_left = CuDNNLSTM(32, return_sequences=False)(reshaped_reversed)

    lstm_top_down = CuDNNLSTM(32, return_sequences=False)(transposed)
    lstm_bottom_up = CuDNNLSTM(32, return_sequences=False)(transposed_reversed)

    merged = Concatenate()([
      lstm_left_right, lstm_right_left, lstm_top_down, lstm_bottom_up])

    #dropout_1 = Dropout(self.kwargs['dropout'])(merged)

    #dense_2   = Dense(7, kernel_regularizer=l2(self.kwargs['l2_regu']))(dropout_1)
    dense_2   = Dense(7)(merged)
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

class StandardLSTM(TopModel):

  def build(self, input_tensor):

    assert len(input_tensor.shape[2:]) == 1

    build_type = self.kwargs['build']
    lstm_units = int(self.kwargs['lstm_units'])
    r_act = self.kwargs['r_act']

    if build_type == 'standard':

        if r_act == 'hard_sigmoid':
            #bn_1 = TimeDistributed(BatchNormalization(momentum=0.99))(input_tensor)
            lstm_1 = Bidirectional(
                              LSTM(lstm_units, return_sequences=True, name='lstm_1'),
                        merge_mode='concat', name='bi_lstm_1')(input_tensor)
        elif r_act == 'tanh':
            lstm_1 = LSTM(lstm_units, recurrent_activation='tanh', return_sequences=True)(input_tensor)

        dense_3   = TimeDistributed(Dense(7, name='dense_7'), name='td_dense_7')(lstm_1)
        quat_norm = TimeDistributed(QuaternionNormalization(name='quat_norm'), name='td_quat_norm')(dense_3)
        return TimeDistributed(Lambda(lambda x: x, name='prediction_inner'), name='prediction')(quat_norm)
        
    elif build_type == 'spatial':
        dense_1 = TimeDistributed(Dense(1024, activation='relu'))(input_tensor)

        reshaped = TimeDistributed(Reshape((32, 32)))(dense_1)
        reshaped_reversed = TimeDistributed(Lambda(lambda x: K.reverse(x, axes=1)))(reshaped)

        transposed = TimeDistributed(Lambda(lambda x: K.permute_dimensions(x, (0,2,1))))(reshaped)
        transposed_reversed = TimeDistributed(Lambda(lambda x: K.reverse(x, axes=1)))(transposed)

        lstm_left_right = TimeDistributed(CuDNNLSTM(32, return_sequences=False))(reshaped)
        lstm_right_left = TimeDistributed(CuDNNLSTM(32, return_sequences=False))(reshaped_reversed)

        lstm_top_down = TimeDistributed(CuDNNLSTM(32, return_sequences=False))(transposed)
        lstm_bottom_up = TimeDistributed(CuDNNLSTM(32, return_sequences=False))(transposed_reversed)

        merged = Concatenate()([
          lstm_left_right, lstm_right_left, lstm_top_down, lstm_bottom_up])
        
        if r_act == 'hard_sigmoid':
            lstm_1 = CuDNNLSTM(lstm_units, return_sequences=True)(merged)
        elif r_act == 'tanh':
            lstm_1 = LSTM(lstm_units, recurrent_activation='tanh', return_sequences=True)(merged)

        dense_3   = TimeDistributed(Dense(7))(lstm_1)
        quat_norm = TimeDistributed(QuaternionNormalization(name='quat_norm'))(dense_3)
        return TimeDistributed(Lambda(lambda x: x), name='prediction')(quat_norm)


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

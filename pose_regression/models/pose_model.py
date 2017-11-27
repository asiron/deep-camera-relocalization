from __future__ import print_function

from keras.models import Sequential, Model
from keras.layers import Input, Add, Lambda, TimeDistributed, Concatenate

import keras.backend as K

import top_models, losses, layers

from ..cnn.vgg16.vgg16 import VGG16
from ..cnn.googlenet.googlenet import GoogleNet
from ..cnn.inception_resnet_v2.inception_resnet_v2 import InceptionResNetV2

class PoseModel(object):

  CNN_MODELS = {
    'vgg16' : VGG16,
    'googlenet' : GoogleNet,
    'inception_resnet_v2' : InceptionResNetV2
  }

  TOPMODELS = {
    'spatial-lstm'  : top_models.SpatialLSTM,
    'standard-lstm' : top_models.StandardLSTM,
    'stateful-lstm' : top_models.StatefulLSTM,
    'regressor'     : top_models.Regressor
  }

  MODES = ['initial', 'finetune']

  def __init__(self, input_shape=None, top_model_type='regressor', 
    model_loss='naive-weighted', mode='initial', finetuning_model_arch=None, 
    finetuning_model_dataset=None, model_weights=None, batch_size=None,
    seq_len=None, subseq_len=None, **hyperparams):
    '''
    Creates a PoseModel, which can be used for initial learning or finetuning

    If mode was set to 'finetune', then finetune_model has to be passed in.
    finetune_model is usually the last convolution or inception block,
    we will finetune. A full keras model has to be passed in (arch+weights)

    input_shape (of the top model) has to be only specified 
    during initial training. During finetuning, the input_shape will be inferred
    from the finetuning model.
    '''
    
    if top_model_type not in PoseModel.TOPMODELS.keys():
      raise ValueError('top_model_type has to be one of {}'
        .format(PoseModel.TOPMODELS.keys()))

    if model_loss not in losses.LOSSES.keys():
      raise ValueError('model_loss has to be one of {}'.format(losses.LOSSES.keys()))

    if mode not in PoseModel.MODES:
      raise ValueError('mode has to be one of {}'.format(PoseModel.MODES))

    if mode == 'finetune' \
      and not (finetuning_model_arch and finetuning_model_dataset):
      raise ValueError(('If mode is set to finetune, architecture ',
        'and dataset have to be given!'))

    if mode == 'initial' and not input_shape:
      raise ValueError('Input shape has to specified during initial training')
    
    if 'stateful' in top_model_type and not (batch_size and subseq_len):
      raise ValueError('Batch size and sub-sequence length have to be ' + 
        'specified in stateful LSTM!')

    if 'standard' in top_model_type and seq_len is None:
      raise ValueError('Sequence length has to be specified in standard LSTM!')


    self.input_shape = input_shape
    self.top_model_type = top_model_type
    self.model_loss = model_loss
    self.mode = mode
    self.finetuning_model_arch = finetuning_model_arch
    self.finetuning_model_dataset = finetuning_model_dataset
    self.model_weights = model_weights
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.subseq_len = subseq_len
    self.hyperparams = hyperparams

  def build(self):

    top_model_class = PoseModel.TOPMODELS[self.top_model_type]

    if self.mode == 'initial':
      main_input = self.make_input_shape(self.input_shape, name='main_input')
      top_model_builder = top_model_class(**self.hyperparams)
      top_model_output = top_model_builder.build(main_input)

    elif self.mode == 'finetune':

      finetune_model = self.make_finetuning_model()
      main_input_shape = finetune_model.input_shape[1:]

      if 'standard' in self.top_model_type or 'stateful' in self.top_model_type:
        finetune_model = TimeDistributed(finetune_model)

      main_input = self.make_input_shape(main_input_shape, name='main_input')
      print(main_input)
      finetune_output = finetune_model(main_input)

      top_model_builder = top_model_class(**self.hyperparams)
      top_model_output = top_model_builder.build(finetune_output)

    if 'homoscedastic' in self.model_loss:
      '''Adding Homoscedastic Loss as last layers to the model'''
      #def loss(y_true, y_pred):
      #  return y_pred

      loss = dummy_loss = losses.DummyLoss()
      labels_input = self.make_input_shape((7,), name='labels_input')
      loss_output = self.add_homoscedastic_loss(top_model_output, labels_input)
      self.model = Model(inputs=[main_input, labels_input], outputs=loss_output)
    else:
      loss = losses.LOSSES[self.model_loss](**self.hyperparams)
      self.model = Model(inputs=[main_input], outputs=top_model_output)

    if self.mode == 'finetune' and self.model_weights:
      print('Loading weights of the model from {}.'.format(self.model_weights))
      self.model.load_weights(self.model_weights, by_name=True) 

    optimizer_builder = self.hyperparams['optimizer']
    optimizer = optimizer_builder(**self.hyperparams)

    self.model.compile(optimizer=optimizer, loss=loss)
    self.model.summary()
    return self.model

  def make_finetuning_model(self):
    finetuning_model_class = PoseModel.CNN_MODELS[self.finetuning_model_arch]
    return finetuning_model_class(
      dataset=self.finetuning_model_dataset, 
      mode='finetune').build()

  def add_homoscedastic_loss(self, top_model_output, labels_input):
    if 'standard' in self.top_model_type or 'stateful' in self.top_model_type:
      merged = Concatenate(axis=-1)([top_model_output, labels_input])
      pq_losses = TimeDistributed(
        losses.LOSSES[self.model_loss](
          name='pq_losses',
          is_lstm=True,
          **self.hyperparams), name='td_pq_losses')(merged)
      p_loss = TimeDistributed(
        Lambda(
          lambda x: K.expand_dims(x[..., 0], axis=-1),
          name='expand_p_loss_dim'),
        name='td_expand_p_loss_dim')(pq_losses)
      q_loss = TimeDistributed(
        Lambda(
          lambda x: K.expand_dims(x[..., 1], axis=-1),
          name='expand_q_loss_dim'),
        name='td_expand_q_loss_dim')(pq_losses)
      p_loss = TimeDistributed(
        layers.HomoscedasticLoss(0,  name='homo_p_loss'),
        name='homo_pos_loss')(p_loss)
      q_loss = TimeDistributed(
        layers.HomoscedasticLoss(-3, name='homo_q_loss'),
        name='homo_quat_loss')(q_loss)
      pq_concat = Concatenate(axis=-1, name='concat_pq_losses')([p_loss, q_loss])
      return TimeDistributed(
          Lambda(
            lambda x: K.sum(x, axis=-1),
            name='loss_output_inner'), 
        name='loss_output')(pq_concat)
    
    else:
      pq_losses = losses.LOSSES[self.model_loss](name='pq_losses',
        **self.hyperparams)([top_model_output, labels_input])

      p_loss = Lambda(lambda x: K.expand_dims(x[..., 0], axis=-1))(pq_losses)
      q_loss = Lambda(lambda x: K.expand_dims(x[..., 1], axis=-1))(pq_losses)
      p_loss = layers.HomoscedasticLoss(0,  name='homo_pos_loss')(p_loss)
      q_loss = layers.HomoscedasticLoss(-3, name='homo_quat_loss')(q_loss)
      return Add(name='loss_output')([p_loss, q_loss]) 

  def make_input_shape(self, shape, name=None):
    if 'stateful' in self.top_model_type:
      batch_shape=(self.batch_size, self.subseq_len) + shape
      return Input(batch_shape=batch_shape, name=name)
    elif 'standard' in self.top_model_type:
      return Input(shape=(self.seq_len,) + shape, name=name)
    else:
      return Input(shape=shape, name=name)

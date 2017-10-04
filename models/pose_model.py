from keras.models import Sequential
from keras.optimizers import Adam

import regressor, lstm, losses

TOPMODELS = {
  'lstm'      : lstm.LSTM,
  'regressor' : regressor.Regressor
}

class PoseModel(object):

  MODES = ['initial', 'finetune']

  def __init__(self, input_shape, top_model_type='regressor', 
    model_loss='naive_weighted', mode='initial', finetune_model=None, 
    topmodel_weights=None, **hyperparams):
    '''
    Creates a PoseModel, which can be used for initial learning or finetuning

    If mode was set to 'finetune', then finetune_model has to be passed in.
    finetune_model is usually the last convolution or inception block,
    we will finetune. A full keras model has to be passed in (arch+weights)

    '''
    if top_model_type not in TOPMODELS.keys():
      raise ValueError('top_model_type has to be one of {}'.format(TOPMODELS.keys()))

    if model_loss not in losses.LOSSES.keys():
      raise ValueError('model_loss has to be one of {}'.format(losses.LOSSES.keys()))

    if mode not in MODES:
      raise ValueError('mode has to be one of {}'.format(MODES))

    if mode == 'finetune' and not (finetune_model and topmodel_weights):
      raise ValueError(('If mode is set to finetune, finetune_model ',
        'and topmodel_weights have to be given!'))

    self.top_model_type = top_model_type
    self.model_loss = model_loss
    self.mode = mode

    self.input_shape = input_shape
    self.hyperparams = hyperparams

    top_model = TOPMODELS[self.top_model_type](self.input_shape, **self.hyperparams)

    if self.mode == 'initial':
      self.model = top_model

    elif self.mode == 'finetune':
      top_model.load_weights(self.topmodel_weights)
      self.model = Sequential()
      self.model.add(self.finetune_model)
      self.model.add(top_model)

    loss = losses.LOSSES[self.model_loss](self.hyperparams)
    optimizer = Adam(lr=self.hyperparams['l_rate'])
    self.model.compile(optimizer=optimizer, loss=loss)
    self.model.summary()

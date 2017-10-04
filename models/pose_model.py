from keras.models import Sequential
from keras.optimizers import Adam

import regressor, lstm, losses

TOPMODELS = {
  'lstm'      : lstm.LSTM,
  'regressor' : regressor.Regressor
}

class PoseModel(object):

  MODES = ['initial', 'finetune']

  def __init__(self, input_shape=None, top_model_type='regressor', 
    model_loss='naive_weighted', mode='initial', finetune_model=None, 
    topmodel_weights=None, **hyperparams):
    '''
    Creates a PoseModel, which can be used for initial learning or finetuning

    If mode was set to 'finetune', then finetune_model has to be passed in.
    finetune_model is usually the last convolution or inception block,
    we will finetune. A full keras model has to be passed in (arch+weights)

    input_shape (of the top model) has to be only specified 
    during initial training. During finetuning, the input_shape will be inferred
    from the finetuning model.

    '''
    if top_model_type not in TOPMODELS.keys():
      raise ValueError('top_model_type has to be one of {}'.format(TOPMODELS.keys()))

    if model_loss not in losses.LOSSES.keys():
      raise ValueError('model_loss has to be one of {}'.format(losses.LOSSES.keys()))

    if mode not in PoseModel.MODES:
      raise ValueError('mode has to be one of {}'.format(PoseModel.MODES))

    if mode == 'finetune' and not (finetune_model and topmodel_weights):
      raise ValueError(('If mode is set to finetune, finetune_model ',
        'and topmodel_weights have to be given!'))

    if mode == 'initial' and not input_shape:
      raise ValueError('Input shape has to specified during initial training')


    self.top_model_type = top_model_type
    self.model_loss = model_loss
    self.mode = mode

    self.hyperparams = hyperparams

    top_model_class = TOPMODELS[self.top_model_type]

    if self.mode == 'initial':

      self.input_shape = input_shape
      self.model = top_model_class(self.input_shape, **self.hyperparams).model

    elif self.mode == 'finetune':

      self.input_shape = self.finetune_model.output_shape
      top_model = top_model_class(self.input_shape, **self.hyperparams).model
      top_model.load_weights(self.topmodel_weights)

      self.model = Sequential()
      self.model.add(self.finetune_model)
      self.model.add(top_model)

    loss = losses.LOSSES[self.model_loss](**self.hyperparams)
    optimizer = Adam(lr=self.hyperparams['l_rate'])
    self.model.compile(optimizer=optimizer, loss=loss)
    self.model.summary()

from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.layers import Input, Add, Lambda, TimeDistributed

import regressor, lstm, losses, layers

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
    
    self.hyperparams = hyperparams
    self.top_model_type = top_model_type
    self.model_loss = model_loss
    self.mode = mode

    top_model_class = TOPMODELS[self.top_model_type]

    if self.mode == 'initial':

      main_input = Input(shape=input_shape) 
      top_model_builder = top_model_class(**self.hyperparams)
      top_model_output = top_model_builder.build(main_input)

    elif self.mode == 'finetune':

      main_input_shape = finetune_model.input_shape[1:]
      main_input = Input(shape=main_input_shape)
      finetune_output = finetune_model(main_input)

      top_model_input_shape = (finetune_model.output_shape[1],)
      top_model_builder = top_model_class(**self.hyperparams)
      top_model_output = top_model_builder.build(finetune_output)

    if 'homoscedastic' in self.model_loss:
      '''Adding Homoscedastic Loss as last layers to the model'''
      
      labels_input = Input(name='labels_input', shape=[7])
      pq_losses = losses.LOSSES[self.model_loss](name='pq_losses',
        **hyperparams)([top_model_output, labels_input])

      p_loss = Lambda(lambda x: x[..., 0])(pq_losses)
      q_loss = Lambda(lambda x: x[..., 1])(pq_losses)
      p_loss = layers.HomoscedasticLoss(0,  name='homo_pos_loss')(p_loss)
      q_loss = layers.HomoscedasticLoss(-3, name='homo_quat_loss')(q_loss)
      loss_output = Add()([p_loss, q_loss])

      self.model = Model(inputs=[main_input, labels_input], outputs=loss_output)
      loss = lambda y_true, y_pred: y_pred

    else:
      self.model = Model(inputs=[main_input], outputs=top_model_output)
      loss = losses.LOSSES[self.model_loss](**self.hyperparams)

    #weights_before = self.model.get_layer('dense_1').get_weights()
    #if self.mode == 'finetune':
    #  self.model.load_weights(topmodel_weights, by_name=True)
    
    #weights_after = self.model.get_layer('dense_1').get_weights()

    #optimizer = SGD(lr=self.hyperparams['l_rate'], momentum=0.9)
    optimizer = Adam(lr=self.hyperparams['l_rate'])
    self.model.compile(optimizer=optimizer, loss=loss)
    self.model.summary()

    #weights_compiled = self.model.get_layer('dense_1').get_weights()

    # print weights_before
    # print weights_after
    # print weights_compiled
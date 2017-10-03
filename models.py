import keras.backend as K
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dropout, Dense, Lambda
from keras.optimizers import Adam
from keras.regularizers import l2

def scope_wrapper(func, *args, **kwargs):
  def scoped_func(*args, **kwargs):
    with tf.name_scope("quat_{}".format(func.__name__)):
      return func(*args, **kwargs)
  return scoped_func

class WeightedPoseLoss(object):

  __name__ = 'abstract_w_pose_loss'

  def __init__(self, beta=200, gamma=1):
    self.beta = beta
    if gamma == 1:
      self.L_gamma = K.abs
    elif gamma == 2:
      self.L_gamma = K.square
    else:
      raise ValueError('gamma has to be either 1 or 2 (L1 or L2 loss)')

  def __call__(self, y_true, y_pred):
    return self.loss(y_true, y_pred)

  def loss(self, y_true, y_pred):
    p_loss = self.position_loss(y_true, y_pred)
    q_loss = self.quaternion_loss(y_true, y_pred)
    return p_loss + q_loss * self.beta    

  def position_loss(self, y_true, y_pred):
    raise NotImplementedError('Position loss has to be implemented!')

  def quaternion_loss(self, y_true, y_pred):
    raise NotImplementedError('Quaternion loss has to be implemented!')

  def L_gamma_loss(self, y_true, y_pred):
    return K.mean(self.L_gamma(y_true - y_pred), axis=-1)

class NaiveWeightedPoseLoss(WeightedPoseLoss):

  __name__ = 'naive_w_pose_loss'

  def position_loss(self, y_true, y_pred):
    pos_true, pos_pred = y_true[..., :3], y_pred[..., :3]
    return self.L_gamma_loss(pos_true, pos_pred)

  def quaternion_loss(self, y_true, y_pred):
    quat_true, quat_pred = y_true[..., 3:], y_pred[..., 3:]
    return self.L_gamma_loss(quat_true, quat_pred)

class ProperWeightedPoseLoss(WeightedPoseLoss):

  __name__ = 'proper_w_pose_loss'

  @scope_wrapper
  def position_loss(self, y_true, y_pred):
    pos_true, pos_pred = y_true[..., :3], y_pred[..., :3]
    return self.L_gamma_loss(pos_true, pos_pred)

  @scope_wrapper
  def quaternion_loss(self, y_true, y_pred):
    quat_true, quat_pred = y_true[..., 3:], y_pred[..., 3:]
    quat_diff = self.quaternion_mul(quat_true, self.quaternion_conj(quat_pred))
    quat_error = quat_diff[..., :3] * 0.5
    return K.mean(self.L_gamma(quat_error), axis=-1)

  @scope_wrapper
  def quaternion_mul(self, q1, q2):
    x1, y1, z1, w1 = tf.unstack(q1, num=4, axis=-1)
    x2, y2, z2, w2 = tf.unstack(q2, num=4, axis=-1)
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return tf.stack([x, y, z, w], axis=-1)

  @scope_wrapper
  def quaternion_conj(self, q):
    q_vector, q_w = q[..., :-1], tf.reshape(q[..., -1], [-1, 1])
    return tf.concat([q_vector, -q_w], axis=-1)

class QuaternionNormalization(Lambda):

  def __init__(self, name=None):
    def layer(x):
      pos, quat = x[..., :3], x[..., 3:]
      quat = K.l2_normalize(quat, axis=-1)
      return K.concatenate([pos, quat], axis=-1)

    super(QuaternionNormalization, self).__init__(layer, name=name)


class WeightedLinearRegression(object):

  def __init__(self, input_shape, **kwargs):

    self.kwargs = kwargs

    self.model = Sequential()
    self.model.add(Dense(2048,
      activation='relu', 
      input_shape=(input_shape,),
      W_regularizer=l2(self.kwargs['l2_regu']),
      name='dense_1'))
    self.model.add(Dropout(self.kwargs['dropout'], name='dropout_1'))
    self.model.add(Dense(7, name='dense_2'))
    self.model.add(QuaternionNormalization(name='quat_norm'))

    self.optimizer = Adam(lr=self.kwargs['l_rate'])

  def build(self):
    raise NotImplementedError('build method must be implemented in subclass!')

class NaiveWeightedLinearRegression(WeightedLinearRegression):

  def build(self):
    loss = NaiveWeightedPoseLoss(
      beta=self.kwargs['beta'],
      gamma=self.kwargs['gamma'])
    self.model.compile(optimizer=self.optimizer, loss=loss)
    self.model.summary()
    return self.model

class ProperWeightedLinearRegression(WeightedLinearRegression):

  def build(self):
    loss = ProperWeightedPoseLoss(
      beta=self.kwargs['beta'],
      gamma=self.kwargs['gamma'])
    self.model.compile(optimizer=self.optimizer, loss=loss)
    return self.model

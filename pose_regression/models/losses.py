import tensorflow as tf
import keras.backend as K

from keras.layers import Lambda, Layer

def scope_wrapper(func, *args, **kwargs):
  def scoped_func(*args, **kwargs):
    with tf.name_scope("quat_{}".format(func.__name__)):
      return func(*args, **kwargs)
  return scoped_func

@scope_wrapper
def quaternion_mul(q1, q2):
  x1, y1, z1, w1 = tf.unstack(q1, num=4, axis=-1)
  x2, y2, z2, w2 = tf.unstack(q2, num=4, axis=-1)
  x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
  y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
  z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
  w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
  return tf.stack([x, y, z, w], axis=-1)

@scope_wrapper
def quaternion_conj(q):
  xyz, w = q[..., :-1], tf.expand_dims(q[..., -1], axis=-1)
  return tf.concat([xyz, -w], axis=-1)

@scope_wrapper
def quaternion_angle_loss(q1, q2, gamma):
  diff = quaternion_mul(q1, quaternion_conj(q2))
  diff = K.clip(diff, -1.0, 1.0)
  xyz, w = diff[..., :-1], tf.expand_dims(diff[..., -1], axis=-1)
  norms = tf.expand_dims(tf.norm(xyz, axis=-1), axis=-1)
  angle = 2 * tf.atan2(norms, K.abs(w))
  print(norms.shape, w.shape, angle.shape)
  return angle ** 2 if gamma == 2 else K.abs(angle)

@scope_wrapper
def quaternion_error_loss(q1, q2, L_gamma):
  diff = quaternion_mul(q1, quaternion_conj(q2))
  return 0.5 * L_gamma(diff[..., :3])

class DummyLoss(object):

  __name__ = '_dummy_loss'

  def __call__(self, y_true, y_pred):    
    return y_pred

class PoseLoss(Layer):

  def __init__(self, name=None, **hyperparams):

    self.is_lstm = hyperparams.get('is_lstm', False)
    self.gamma = int(hyperparams.get('gamma', -1))
    if self.gamma == 1:
      self.L_gamma = lambda x: K.mean(K.abs(x), axis=-1)
    elif self.gamma == 2:
      self.L_gamma = lambda x: K.mean(K.square(x), axis=-1)
    else:
      raise ValueError('gamma has to be either 1 or 2 (L1 or L2 loss)')

    super(PoseLoss, self).__init__(name=name)

  def build(self, input_shape):
    super(PoseLoss, self).build(input_shape)

  def call(self, x):
    if self.is_lstm:
      y_pred, y_true = x[..., :7], x[..., 7:]
    else:
      y_pred, y_true = x

    pos_true,  pos_pred  = y_true[..., :3], y_pred[..., :3]
    quat_true, quat_pred = y_true[..., 3:], y_pred[..., 3:]
    p_loss = self.position_loss(pos_true, pos_pred)
    q_loss = self.quaternion_loss(quat_true, quat_pred)
    p_loss_1 = K.reshape(p_loss, [-1, 1])
    q_loss_1 = K.reshape(q_loss, [-1, 1])
    return K.concatenate([p_loss_1, q_loss_1], axis=-1)

  def compute_output_shape(self, input_shape):
    if isinstance(input_shape, list):
      return map(lambda x: x[:-1] + (2,), input_shape)
    else:
      return input_shape[:-1] + (2,)

  def get_config(self):
    config = {'gamma': self.gamma, 'is_lstm': self.is_lstm}
    base_config = super(PoseLoss, self).get_config()
    new_config = dict(list(base_config.items()) + list(config.items()))
    return new_config

  def position_loss(self, pos_true, pos_pred):
    raise NotImplementedError('Position loss has to be implemented!')

  def quaternion_loss(self, quat_true, quat_pred):
    raise NotImplementedError('Quaternion loss has to be implemented!')

  def L_gamma_loss(self, err):
    return self.L_gamma(err)

class NaivePoseLoss(PoseLoss):

  def position_loss(self, pos_true, pos_pred):
    return self.L_gamma_loss(pos_true - pos_pred)

  def quaternion_loss(self, quat_true, quat_pred):
    return self.L_gamma_loss(quat_true - quat_pred)

class QuaternionErrorPoseLoss(PoseLoss):

  def __init__(self, name=None, **kwargs):
    super(QuaternionErrorPoseLoss, self).__init__(name=name, **kwargs)

  def position_loss(self, pos_true, pos_pred):
    return self.L_gamma_loss(pos_true - pos_pred)

  def quaternion_loss(self, quat_true, quat_pred):
    return quaternion_error_loss(quat_true, quat_pred, self.L_gamma)
    
class QuaternionAnglePoseLoss(PoseLoss):

  def position_loss(self, pos_true, pos_pred):
    return self.L_gamma_loss(pos_true - pos_pred)

  def quaternion_loss(self, quat_true, quat_pred):
    return quaternion_angle_loss(quat_true, quat_pred, self.gamma)

class WeightedPoseLoss(object): 

  __name__ = 'abstract_weighted_pose_loss'

  def __init__(self, **hyperparams):

    self.beta = float(hyperparams.get('beta', 200))
    self.gamma = int(hyperparams.get('gamma', -1))

    self.gamma = hyperparams.get('gamma', -1)
    if self.gamma == 1:
      self.L_gamma = lambda x: K.mean(K.abs(x), axis=-1)
    elif self.gamma == 2:
      self.L_gamma = lambda x: K.mean(K.square(x), axis=-1)
    else:
      raise ValueError('gamma has to be either 1 or 2 (L1 or L2 loss)')

  def __call__(self, y_true, y_pred):
    return self.loss(y_true, y_pred)

  def loss(self, y_true, y_pred):
    pos_true,  pos_pred  = y_true[..., :3], y_pred[..., :3]
    quat_true, quat_pred = y_true[..., 3:], y_pred[..., 3:]
    p_loss = self.position_loss(pos_true, pos_pred)
    q_loss = self.quaternion_loss(quat_true, quat_pred)
    return p_loss + q_loss * self.beta

  def position_loss(self, pos_true, pos_pred):
    raise NotImplementedError('Position loss has to be implemented!')

  def quaternion_loss(self, quat_true, quat_pred):
    raise NotImplementedError('Quaternion loss has to be implemented!')

  def L_gamma_loss(self, err):
    return self.L_gamma(err)

class NaiveWeightedPoseLoss(WeightedPoseLoss):

  __name__ = 'naive_weighted_pose_loss'

  def position_loss(self, pos_true, pos_pred):
    return self.L_gamma_loss(pos_true - pos_pred)

  def quaternion_loss(self, quat_true, quat_pred):
    return self.L_gamma_loss(quat_true - quat_pred)

class QuaternionErrorWeightedPoseLoss(WeightedPoseLoss):

  __name__ = 'quaternion_err_weighted_pose_loss'

  def position_loss(self, pos_true, pos_pred):
    return self.L_gamma_loss(pos_true - pos_pred)

  def quaternion_loss(self, quat_true, quat_pred):
    return quaternion_error_loss(quat_true, quat_pred, self.L_gamma)

class QuaternionAngleWeightedPoseLoss(WeightedPoseLoss):

  __name__ = 'quaternion_angle_weighted_pose_loss'

  def position_loss(self, pos_true, pos_pred):
    return self.L_gamma_loss(pos_true - pos_pred)

  def quaternion_loss(self, quat_true, quat_pred):
    return quaternion_angle_loss(quat_true, quat_pred, self.gamma)


class PositionOnlyLoss(object): 

  __name__ = '_position_only_loss'

  def __init__(self, **hyperparams):

    self.gamma = int(hyperparams.get('gamma', -1))
    if self.gamma == 1:
      self.L_gamma = lambda x: K.mean(K.abs(x), axis=-1)
    elif self.gamma == 2:
      self.L_gamma = lambda x: K.mean(K.square(x), axis=-1)
    else:
      raise ValueError('gamma has to be either 1 or 2 (L1 or L2 loss)')

  def __call__(self, y_true, y_pred):
    return self.loss(y_true, y_pred)

  def loss(self, y_true, y_pred):
    pos_true, pos_pred  = y_true[..., :3], y_pred[..., :3]
    p_loss = self.position_loss(pos_true, pos_pred)
    return p_loss

  def position_loss(self, pos_true, pos_pred):
    return self.L_gamma(pos_true - pos_pred)

  def L_gamma_loss(self, err):
    return self.L_gamma(err)

class QuaternionOnlyLoss(object): 

  __name__ = '_quaternion_only_loss'

  def __init__(self, **hyperparams):

    self.gamma = hyperparams.get('gamma', -1)
    if self.gamma == 1:
      self.L_gamma = lambda x: K.mean(K.abs(x), axis=-1)
    elif self.gamma == 2:
      self.L_gamma = lambda x: K.mean(K.square(x), axis=-1)
    else:
      raise ValueError('gamma has to be either 1 or 2 (L1 or L2 loss)')

  def __call__(self, y_true, y_pred):
    return self.loss(y_true, y_pred)

  def loss(self, y_true, y_pred):
    quat_true, quat_pred = y_true[..., 3:], y_pred[..., 3:]
    q_loss = self.quaternion_loss(quat_true, quat_pred)
    return q_loss

  def quaternion_loss(self, quat_true, quat_pred):
    return quaternion_angle_loss(quat_true, quat_pred, self.gamma)

  def L_gamma_loss(self, err):
    return self.L_gamma(err)

LOSSES = {
  'naive-weighted' : NaiveWeightedPoseLoss,
  'quaternion-error-weighted' : QuaternionErrorWeightedPoseLoss,
  'quaternion-angle-weighted' : QuaternionAngleWeightedPoseLoss,

  'naive-homoscedastic': NaivePoseLoss,
  'quaternion-error-homoscedastic' : QuaternionErrorPoseLoss,
  'quaternion-angle-homoscedastic' : QuaternionAnglePoseLoss,

  'only-position' : PositionOnlyLoss,
  'only-quaternion' : QuaternionOnlyLoss
}

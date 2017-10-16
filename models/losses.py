import tensorflow as tf
import keras.backend as K

from keras.layers import Lambda

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
  q_vector, q_w = q[..., :-1], tf.reshape(q[..., -1], [-1, 1])
  return tf.concat([q_vector, -q_w], axis=-1)

class PoseLoss(Lambda):

  def __init__(self, name=None, **hyperparams):

    self.gamma = hyperparams.get('gamma', -1)
    if self.gamma == 1:
      self.L_gamma = lambda x: K.sum(K.abs(x), axis=-1)
    elif self.gamma == 2:
      self.L_gamma = lambda x: K.sqrt(K.sum(K.square(x), axis=-1))
    else:
      raise ValueError('gamma has to be either 1 or 2 (L1 or L2 loss)')

    def loss(args):
      y_pred, y_true = args
      pos_true,  pos_pred  = y_true[..., :3], y_pred[..., :3]
      quat_true, quat_pred = y_true[..., 3:], y_pred[..., 3:]
      p_loss = self.position_loss(pos_true, pos_pred)
      q_loss = self.quaternion_loss(quat_true, quat_pred)
      p_loss = K.reshape(p_loss, [-1, 1])
      q_loss = K.reshape(q_loss, [-1, 1])
      return K.concatenate([p_loss, q_loss], axis=-1)

    super(PoseLoss, self).__init__(loss, output_shape=(2,), name=name)

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

  def position_loss(self, pos_true, pos_pred):
    return self.L_gamma_loss(pos_true - pos_pred)

  def quaternion_loss(self, quat_true, quat_pred):
    quat_diff = quaternion_mul(quat_true, quaternion_conj(quat_pred))
    quat_error = quat_diff[..., :3]
    return 0.5 * self.L_gamma(quat_error)

class QuaternionAnglePoseLoss(PoseLoss):

  def position_loss(self, pos_true, pos_pred):
    return self.L_gamma_loss(pos_true - pos_pred)

  def quaternion_loss(self, quat_true, quat_pred):
    quat_diff = quaternion_mul(quat_true, quaternion_conj(quat_pred))
    xyz, w = quat_diff[..., :3], quat_diff[..., 3]
    return K.abs(2*tf.atan2(tf.norm(xyz, axis=-1), w))

class WeightedPoseLoss(object):

  __name__ = 'abstract_weighted_pose_loss'

  def __init__(self, **hyperparams):

    self.beta = hyperparams.get('beta', 200)
    self.gamma = hyperparams.get('gamma', -1)

    self.gamma = hyperparams.get('gamma', -1)
    if self.gamma == 1:
      self.L_gamma = lambda x: K.sum(K.abs(x), axis=-1)
    elif self.gamma == 2:
      self.L_gamma = lambda x: K.sqrt(K.sum(K.square(x), axis=-1))
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
    quat_diff = quaternion_mul(quat_true, quaternion_conj(quat_pred))
    quat_error = quat_diff[..., :3]
    return 0.5 * self.L_gamma(quat_error)

class QuaternionAngleWeightedPoseLoss(WeightedPoseLoss):

  __name__ = 'quaternion_angle_weighted_pose_loss'

  def position_loss(self, pos_true, pos_pred):
    return self.L_gamma_loss(pos_true - pos_pred)

  def quaternion_loss(self, quat_true, quat_pred):
    quat_diff = quaternion_mul(quat_true, quaternion_conj(quat_pred))
    xyz, w = quat_diff[..., :3], quat_diff[..., 3]
    return K.abs(2 * tf.atan2(tf.norm(xyz, axis=-1), w))

LOSSES = {
  'naive_weighted' : NaiveWeightedPoseLoss,
  'quaternion_error_weighted' : QuaternionErrorWeightedPoseLoss,
  'quaternion_angle_weighted' : QuaternionAngleWeightedPoseLoss,

  'naive_homoscedastic': NaivePoseLoss,
  'quaternion_error_homoscedastic' : QuaternionErrorPoseLoss,
  'quaternion_angle_homoscedastic' : QuaternionAnglePoseLoss
}

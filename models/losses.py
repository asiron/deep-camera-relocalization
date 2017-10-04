import tensorflow as tf
import keras.backend as K

def scope_wrapper(func, *args, **kwargs):
  def scoped_func(*args, **kwargs):
    with tf.name_scope("quat_{}".format(func.__name__)):
      return func(*args, **kwargs)
  return scoped_func

class WeightedPoseLoss(object):

  __name__ = 'abstract_w_pose_loss'

  def __init__(self, **hyperparams):

    self.beta = hyperparams.get('beta', 200)
    self.gamma = hyperparams.get('gamma', -1)

    if self.gamma == 1:
      self.L_gamma = K.abs
    elif self.gamma == 2:
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

  __name__ = 'naive_weighted_pose_loss'

  def position_loss(self, y_true, y_pred):
    pos_true, pos_pred = y_true[..., :3], y_pred[..., :3]
    return self.L_gamma_loss(pos_true, pos_pred)

  def quaternion_loss(self, y_true, y_pred):
    quat_true, quat_pred = y_true[..., 3:], y_pred[..., 3:]
    return self.L_gamma_loss(quat_true, quat_pred)

class QuaternionWeightedPoseLoss(WeightedPoseLoss):

  __name__ = 'quaternion_weighted_pose_loss'

  def position_loss(self, y_true, y_pred):
    pos_true, pos_pred = y_true[..., :3], y_pred[..., :3]
    return self.L_gamma_loss(pos_true, pos_pred)

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

LOSSES = {
  'naive_weighted' : NaiveWeightedPoseLoss,
  'quaternion_weighted' : QuaternionWeightedPoseLoss,
  'naive_homoscedastic': None,
  'quaternion_homoscedastic' : None
}

naive_weighted = nw = NaiveWeightedPoseLoss
quaternion_weighted = qw = QuaternionWeightedPoseLoss

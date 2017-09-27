import tensorflow as tf
import keras.backend as K

from itertools import izip
from functools32 import lru_cache

from keras.models import Sequential
from keras.layers import Dropout, Dense, Lambda
from keras.optimizers import Adam
from keras.regularizers import l2

from numpy.core.umath_tests import inner1d

class WeightedPoseLoss(object):

  self.__name__ = 'abstract_w_pose_loss'

  def __init__(self, beta = 200, gamma = 1):
    self.beta = beta
    if gamma == 1:
      self.L_gamma = K.abs
    elif gamma == 2:
      self.L_gamma = K.square
    else:
      raise ValueError('gamma has to be either 1 or 2 (L1 or L2 loss)')

  def __call__(self, y_true, y_pred):
    p_loss = self.position_loss(y_true, y_pred)
    q_loss = self.quaternion_loss(y_true, y_pred)
    return p_loss + q_loss * self.beta

  def position_loss(self, y_true, y_pred):
    raise NotImplementedError('Position loss has to be overriden!')

  def quaternion_loss(self, y_true, y_pred):
    raise NotImplementedError('Quaternion loss has to be overriden!')

  def L_gamma_loss(self, y_true, y_pred):
    return K.mean(self.L_gamma(y_true - y_pred), axis=-1)

class NaiveWeightedPoseLoss(WeightedPoseLoss):

  self.__name__ = 'naive_w_pose_loss'

 def position_loss(self, y_true, y_pred):
    pos_true, pos_pred = y_true[..., :3], y_pred[..., :3]
    return self.L_gamma_loss(pos_true, pos_pred)

  def quaternion_loss(self, y_true, y_pred):
    quat_true, quat_pred = y_true[..., 3:], y_pred[..., 3:]
    return self.L_gamma_loss(quat_true, quat_pred)

class ProperWeightedPoseLoss(WeightedPoseLoss):

  self.__name__ = 'proper_w_pose_loss'

 def position_loss(self, y_true, y_pred):
    pos_true, pos_pred = y_true[..., :3], y_pred[..., :3]
    return self.L_gamma_loss(pos_true, pos_pred)

  def quaternion_loss(self, y_true, y_pred):
    quat_true, quat_pred = y_true[..., 3:], y_pred[..., 3:]

    
    return self.L_gamma_loss(quat_true, quat_pred)

class PoseMetrics(object):

  def get_all_metrics():
    return {
      'rmse_position'   : PoseMetrics.rmse_position,
      'mae_position'    : PoseMetrics.mae_position,
      'median_position' : PoseMetrics.median_position,
      'stddev_position' : PoseMetrics.stddev_position,
      'min_position'    : PoseMetrics.min_position,
      'max_position'    : PoseMetrics.max_position,

      'rmse_orientation'   : PoseMetrics.rmse_orientation,
      'mae_orientation'    : PoseMetrics.mae_orientation,
      'median_orientation' : PoseMetrics.median_orientation,
      'stddev_orientation' : PoseMetrics.stddev_orientation,
      'min_orientation'    : PoseMetrics.min_orientation,
      'max_orientation'    : PoseMetrics.max_orientation
    }

  @staticmethod
  def rmse_position(y_true, y_pred):
    errors = PoseMetrics.errors_position(y_true, y_pred)
    return np.sqrt(np.mean(errors ** 2))

  @staticmethod
  def mae_position(y_true, y_pred):
    errors = PoseMetrics.errors_position(y_true, y_pred)
    return np.mean(np.abs(errors))

  @staticmethod
  def median_position(y_true, y_pred):
    errors = PoseMetrics.errors_position(y_true, y_pred)
    return np.median(errors)

  @staticmethod
  def stddev_position(y_true, y_pred):
    errors = PoseMetrics.errors_position(y_true, y_pred)
    return np.std(errors)

  @staticmethod
  def min_position(y_true, y_pred):
    errors = PoseMetrics.errors_position(y_true, y_pred)
    return np.min(errors)

  @staticmethod
  def max_position(y_true, y_pred):
    errors = PoseMetrics.errors_position(y_true, y_pred)
    return np.max(errors)



  @staticmethod
  def rmse_orientation(y_true, y_pred):
    errors = PoseMetrics.errors_orienation(y_true, y_pred)
    return np.sqrt(np.mean(errors ** 2))

  @staticmethod
  def mae_orientation(y_true, y_pred):
    errors = PoseMetrics.errors_orienation(y_true, y_pred)
    return np.mean(np.abs(errors))

  @staticmethod
  def median_orientation(y_true, y_pred):
    errors = PoseMetrics.errors_orienation(y_true, y_pred)
    return np.median(errors)

  @staticmethod
  def stddev_orientation(y_true, y_pred):
    errors = PoseMetrics.errors_orienation(y_true, y_pred)
    return np.std(errors)

  @staticmethod
  def min_orientation(y_true, y_pred):
    errors = PoseMetrics.errors_orienation(y_true, y_pred)
    return np.min(errors)

  @staticmethod
  def max_orientation(y_true, y_pred):
    errors = PoseMetrics.errors_orienation(y_true, y_pred)
    return np.max(errors)

  @lru_cache
  @staticmethod
  def errors_position(y_true, y_pred):
    pos_true, pos_pred = y_true[..., :3], y_pred[..., :3]
    return np.linalg.norm(y_true - y_pred, axis=-1)

  @lru_cache
  @staticmethod
  def errors_orienation(y_true, y_pred):
    quat_true = [quat.Quaternion(quat) for quat in y_true[..., 3:]]
    quat_pred = [quat.Quaternion(quat) for quat in y_pred[..., 3:]]
    return [(q1.inverse * q2).degrees for q1, q2 in izip(quat_true, quat_pred)]

def quaternion_normalize_layer():
  def layer(x):
    pos, quat = x[..., :3], x[..., 3:]
    quat = K.l2_normalize(quat, axis=-1)
    return K.concatenate([pos, quat], axis=-1)
  return Lambda(layer):

def naive_linear_regression(input_shape, dropout=0.5, l_rate=1e-3, 
  beta=200, gamma=1, l2_regu=0.1):

  model = Sequential()
  model.add(Dense(2048,
    activation='relu', 
    input_shape=(input_shape,),
    W_regularizer=l2(l2_regu))
  )
  model.add(Dropout(dropout))
  model.add(Dense(7))
  model.add(Lambda(normalize_quaternion))

  loss = NaiveWeightedPoseLoss(beta=beta, gamma=gamma)
  optimizer = Adam(lr=l_rate)
  #metrics = [PoseMetrics.position_rmse, PoseMetrics.orientation_rmse]
  
  model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
  return model

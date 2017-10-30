import quaternion 
import numpy as np

class PoseMetrics(object):

  @staticmethod
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
    errors = PoseMetrics.abs_errors_position(y_true, y_pred)
    return np.sqrt(np.mean(errors ** 2))

  @staticmethod
  def mae_position(y_true, y_pred):
    errors = PoseMetrics.abs_errors_position(y_true, y_pred)
    return np.mean(np.abs(errors))

  @staticmethod
  def median_position(y_true, y_pred):
    errors = PoseMetrics.abs_errors_position(y_true, y_pred)
    return np.median(errors)

  @staticmethod
  def stddev_position(y_true, y_pred):
    errors = PoseMetrics.abs_errors_position(y_true, y_pred)
    return np.std(errors)

  @staticmethod
  def min_position(y_true, y_pred):
    errors = PoseMetrics.abs_errors_position(y_true, y_pred)
    return np.min(errors)

  @staticmethod
  def max_position(y_true, y_pred):
    errors = PoseMetrics.abs_errors_position(y_true, y_pred)
    return np.max(errors)


  @staticmethod
  def rmse_orientation(y_true, y_pred):
    errors = PoseMetrics.abs_errors_orienation(y_true, y_pred)
    return np.sqrt(np.mean(errors ** 2))

  @staticmethod
  def mae_orientation(y_true, y_pred):
    errors = PoseMetrics.abs_errors_orienation(y_true, y_pred)
    return np.mean(errors)

  @staticmethod
  def median_orientation(y_true, y_pred):
    errors = PoseMetrics.abs_errors_orienation(y_true, y_pred)
    return np.median(errors)

  @staticmethod
  def stddev_orientation(y_true, y_pred):
    errors = PoseMetrics.abs_errors_orienation(y_true, y_pred)
    return np.std(errors)

  @staticmethod
  def min_orientation(y_true, y_pred):
    errors = PoseMetrics.abs_errors_orienation(y_true, y_pred)
    return np.min(errors)

  @staticmethod
  def max_orientation(y_true, y_pred):
    errors = PoseMetrics.abs_errors_orienation(y_true, y_pred)
    return np.max(errors)

  @staticmethod
  def abs_errors_position(y_true, y_pred):
    pos_true, pos_pred = y_true[..., :3], y_pred[..., :3]
    return np.linalg.norm(pos_true - pos_pred, axis=-1)
  
  @staticmethod
  def abs_errors_orienation(y_true, y_pred):
    quat_true = quaternion.as_quat_array(y_true[..., [6,3,4,5]])
    quat_pred = quaternion.as_quat_array(y_pred[..., [6,3,4,5]])
    errors = (quat_true * quat_pred.conjugate())
    errors = quaternion.as_float_array(errors)
    errors_vec, errors_w = errors[:, 1:], errors[:, 0]
    norms = np.linalg.norm(errors_vec, axis=-1)
    angles = np.arctan2(norms, np.abs(errors_w))    
    return np.degrees(2 * angles)

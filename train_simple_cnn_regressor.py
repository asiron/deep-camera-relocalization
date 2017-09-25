#!/usr/bin/env
import argparse, os
import numpy as np

import tensorflow as tf
import keras.backend as K

from keras.models import Sequential
from keras.layers import Dropout, Dense, Lambda
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import (
  LearningRateScheduler, 
  TensorBoard,
  ModelCheckpoint
)

from sklearn.model_selection import train_test_split

from utils import find_files

LABEL_PATTERN = 'pos_[0-9]*.txt'

class NaiveWeightedPoseLoss():

  def __init__(self, beta = 200, gamma = 1):
    self.beta = beta
    self.__name__ = 'naive_w_pose_loss'
    if gamma == 1:
      self.L_gamma = K.abs
    elif gamma == 2:
      self.L_gamma = K.square
    else:
      raise ValueError('gamma has to be either 1 or 2 (L1 or L2 loss)')

  def __call__(self, y_true, y_pred):
    return self.naive_weighted_pose_loss(y_true, y_pred)

  def naive_weighted_pose_loss(self, y_true, y_pred):

    p_loss = self.position_loss(y_true, y_pred)
    q_loss = self.quaternion_loss(y_true, y_pred)

    return p_loss + q_loss * self.beta

  def position_loss(self, y_true, y_pred):
    pos_true, pos_pred = y_true[..., :3], y_pred[..., :3]
    return K.mean(self.L_gamma(pos_true - pos_pred), axis=-1)

  def quaternion_loss(self, y_true, y_pred):
    quat_true, quat_pred = y_true[..., 3:], y_pred[..., 3:]
    return K.mean(self.L_gamma(quat_true - quat_pred), axis=-1)

def position_rmse(y_true, y_pred):
  pos_true, pos_pred = y_true[..., :3], y_pred[..., :3]
  return K.sqrt(K.mean(K.square(pos_true - pos_pred), axis=-1))

def orientation_rmse(y_true, y_pred):
  quat_true, quat_pred = y_true[..., 3:], y_pred[..., 3:]
  diffs = K.abs(K.batch_dot(quat_true, quat_pred, axis=-1))
  diffs = (2 * tf.acos(d)) * 180 / np.pi
  return K.sqrt(K.mean(K.square(diffs)))


def make_model(input_shape, dropout=0.5, l_rate=1e-3, 
  beta=200, gamma=1, l2_regu=0.1):

  def normalize_quaternion(x):
    pos, quat = x[..., :3], x[..., 3:]
    quat = K.l2_normalize(quat, axis=-1)
    return K.concatenate([pos, quat], axis=-1)

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
  metrics = [position_rmse, orientation_rmse]
  
  model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
  return model

def load_labels(directory, pattern=LABEL_PATTERN):
  labels = []
  label_files = find_files(directory, pattern)
  for label_file in label_files:
    with open(label_file, 'r') as file:
      parsed_line = map(float, file.readlines()[0].split(','))
      pose = parsed_line[1:]
      labels.append(pose)

  print 'Labels loaded!'
  return np.array(labels)

def hyperparam_search(X_train, y_train, iters=50):

  gamma_space   = lambda: np.random.randint(1, high=3)
  beta_space    = lambda: np.exp(np.random.uniform(4, 6.5))
  l_rate_space  = lambda: 10 ** np.random.uniform(-6, -3)
  dropout_space = lambda: np.random.uniform(0.1, 0.7)
  l2_regu_space = lambda: np.random.uniform(0.1, 0.4)

  hyperparam_space = {
    'gamma'   : gamma_space, 
    'beta'    : beta_space, 
    'l_rate'  : l_rate_space, 
    'dropout' : dropout_space,
    'l2_regu' : l2_regu_space
  }

  for iteration in xrange(iters):

    sample = {v: space() for v, space in hyperparam_space.items()}

    hyperparam_desc = ('L{gamma},beta={beta:.1f},lr={l_rate:.2e},'
      + 'dropout={dropout:.2f},l2_regu={l2_regu:.2f}').format(**sample)

    lrscheduler = LearningRateScheduler(
      lambda e: l_rate * (0.9 ** (e//80))
    )
    tb_directory = os.path.join(args.output, 'tensorboard')
    tensorboard = TensorBoard(
      log_dir=tb_directory,
      write_graph=True,
      )
    mc_pattern = 'weights.{epoch:04d}-{val_loss:.4f}.hdf5'
    mc_directory = os.path.join(
      args.output, 
      'checkpoints',
      hyperparam_desc,
      mc_pattern
    )
    model_checkpoint = ModelCheckpoint(mc_directory, period=5)
    callbacks = [tensorboard, model_checkpoint, lrscheduler]

    X_train, X_val, y_train, y_val = train_test_split(
      X_train, y_train, test_size=0.75
    )

    model = make_model(2048, **sample)
    model.fit(X, y, 
      batch_size=128,
      validation_data=(X_val, y_val),
      epochs=1000,
      callbacks=callbacks,
      verbose=True
    )

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-l', '--labels', nargs='+', required=True, 
    help='Path to a directory with labels')
  parser.add_argument('-f', '--features', nargs='+', required=True,
    help='Path to a numpy array with features')
  parser.add_argument('-o', '--output', required=True, 
    help='Path to an output dir with tensorboard logs, csv, checkpoints, etc')
  parser.add_argument('-i', '--iters', type=int, default=50,
    help='Number of iterations for the random hyperparameter search')
  args = parser.parse_args()
  
  features = np.array([np.load(f) for f in args.features])
  labels = np.array([load_labels(l) for l in args.labels])

  X_train, X_test, y_train, y_test = train_test_split(
    features, labels, train_size=0.8
  )

  hyperparam_search(X_train, y_train, iters=args.iters)

  # predictions = model.predict(X_valid, verbose=True)
  # np.save('predicted.npy', predictions)
  # np.save('true.npy', y_valid)

if __name__ == '__main__':
  main()
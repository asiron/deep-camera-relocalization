#!/usr/bin/env
import argparse, os
import numpy as np

from keras.callbacks import (
  LearningRateScheduler, 
  ModelCheckpoint,
  EarlyStopping
)

from utils import find_files, make_dir, load_labels, ExtendedLogger
from sklearn.model_selection import train_test_split

from models import pose_model, losses, metrics

def hyperparam_search(model_class, X, y, output=None, iters=50):

  if not output:
    raise ValueError('Output directory has to be defined!')

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

  for _ in xrange(iters):

    sample = {v: space() for v, space in hyperparam_space.items()}

    hyperparam_desc = ('L{gamma},beta={beta:.1f},lr={l_rate:.2e},'
      + 'dropout={dropout:.2f},l2_regu={l2_regu:.2f}').format(**sample)

    #early_stopper = EarlyStopping(monitor='val_loss', 
    #  min_delta=1.0, patience=50, verbose=1)
 
    lrscheduler = LearningRateScheduler(
      lambda e: sample['l_rate'] * (0.9 ** (e//80))
    )

    tb_directory  = os.path.join(output, 'tensorboard', hyperparam_desc)
    csv_directory = os.path.join(output, 'csv', hyperparam_desc)

    logger = ExtendedLogger('quat_norm',
      csv_dir=csv_directory, tb_dir=tb_directory)
    logger.add_validation_metrics(metrics.PoseMetrics.get_all_metrics())

    mc_directory = os.path.join(
      output, 
      'checkpoints',
      hyperparam_desc,
      'weights.{epoch:04d}-{val_loss:.4f}.hdf5'
    )
    make_dir(mc_directory)
    model_checkpoint = ModelCheckpoint(
      mc_directory, 
      save_best_only=True,
      period=10
    )

    #callbacks = [logger, model_checkpoint, lrscheduler, early_stopper]
    callbacks = [logger, model_checkpoint, lrscheduler]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)

    model = model_class(2048, **sample).build()
    model.fit(X_train, y_train, 
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
  
  parser.add_argument('-tm', '--top-model', type=str, default='regressor',
    choices=pose_model.TOPMODELS.keys(),
    help='Top model to use for regression')
  
  parser.add_argument('--loss', type=str, default='naive_weighted',
    choices=losses.LOSSES.keys(),
    help='Loss function to use for optimization')

  parser.add_argument('-m', '--model', type=str, default='initial',
    choices=pose_model.PoseModel.MODES,
    help='Training mode, initial or finetuning')

  parser.add_argument('-i', '--iters', type=int, default=50,
    help='Number of iterations for the random hyperparameter search')


  #parser.add_argument('-c', '--checkpoint', type=str)
  args = parser.parse_args()
  
  if args.mode == 'initial':
    pass
    # model = pose_model.PoseModel()

  elif args.mode == 'finetune':
    pass


  '''

  features = np.vstack([np.load(f) for f in args.features])
  #labels = np.vstack([load_labels(l) for l in args.labels])
  labels = [load_labels(l)[33:] for l in args.labels][0]

  chunk_size = 59
  chunks = [labels[x:x+chunk_size] for x in xrange(0, len(labels), chunk_size)][:10]

  val_chunks = [c[:12] for c in chunks]
  train_chunks = [c[12:47] for c in chunks]
  test_chunks = [c[47:] for c in chunks]

  # X_train, X_test, y_train, y_test = train_test_split(
  #   features, labels, train_size=0.8
  # )

  import matplotlib as mpl
  from mpl_toolkits.mplot3d import Axes3D
  import matplotlib.pyplot as plt

  fig = plt.figure()
  ax = fig.gca(projection='3d')


  for vc in val_chunks:
    ax.plot(vc[:, 0], vc[:, 1], vc[:, 2], color='red')

  for trc in train_chunks:
    ax.plot(trc[:, 0], trc[:, 1], trc[:, 2], color='blue')

  for tec in test_chunks:
    ax.plot(tec[:, 0], tec[:, 1], tec[:, 2], color='green')

  ax.legend()

  plt.show()

  return

  if args.checkpoint:
    from keras.models import load_model
    from models import QuaternionNormalization, ProperWeightedPoseLoss

    model = load_model(args.checkpoint, custom_objects={
      'QuaternionNormalization' : lambda *args, **kwargs: QuaternionNormalization('quat_norm'),
      'proper_w_pose_loss' : ProperWeightedPoseLoss(beta=11, gamma=1)
    })

    model.predict(features, batch_size=128, verbose=1)
    print 'OK!'
    return
  
  if args.model in ['homo-w', 'homo-p']:
    raise NotImplementedError('Homoescedastic uncertaintity is not yet implemented')

  if args.model == 'naive-w':
    model_class = NaiveWeightedLinearRegression
  elif args.model == 'naive-p':
    model_class = ProperWeightedLinearRegression

  hyperparam_search(model_class, X_train, y_train, output=args.output, iters=args.iters)

  # predictions = model.predict(X_valid, verbose=True)
  # np.save('predicted.npy', predictions)
  # np.save('true.npy', y_valid)
  '''

if __name__ == '__main__':
  main()
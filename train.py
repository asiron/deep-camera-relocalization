#!/usr/bin/env
import numpy as np
import argparse, os, importlib

from keras.callbacks import (
  LearningRateScheduler, 
  ModelCheckpoint,
  EarlyStopping
)

from keras.models import load_model

from utils import make_dir, load_labels, ExtendedLogger, make_sequences
from sklearn.model_selection import train_test_split

from models import pose_model, losses, metrics

from cnn.googlenet.googlenet import GoogleNet
from cnn.inception_resnet_v2.inception_resnet_v2 import InceptionResNetV2

FINETUNING_MODELS = {
  'googlenet' : GoogleNet,
  'inception_resnet_v2' : InceptionResNetV2
}

def create_callbacks(output='/tmp', prediction_layer=None, 
  run_identifier=None, l_rate_scheduler=None, save_period=1):
  '''
  Custom logger runs prediction at the end of a training epoch 
  for the validationdataset. In order to retrieve the prediction 
  on a specific layer, we have to pass the names of these layers. 
  This allows for custom loss functions with trainable parameters.
  @see losses
  '''
  early_stopper = EarlyStopping(monitor='val_loss', 
    min_delta=0.0, patience=10, verbose=1)

  lrscheduler = LearningRateScheduler(l_rate_scheduler)

  tb_directory  = os.path.join(output, 'tensorboard', run_identifier)
  csv_directory = os.path.join(output, 'csv', run_identifier)

  logger = ExtendedLogger(prediction_layer,
    csv_dir=csv_directory, tb_dir=tb_directory)
  logger.add_validation_metrics(metrics.PoseMetrics.get_all_metrics())

  mc_directory = os.path.join(
    output,
    'checkpoints',
    run_identifier
  )
  make_dir(mc_directory)
  checkpoint_pattern = 'weights.{epoch:04d}-{val_loss:.4f}.hdf5'
  checkpoint_path = os.path.join(mc_directory, checkpoint_pattern)
  model_checkpoint = ModelCheckpoint(
    checkpoint_path,
    save_weights_only=True,
    save_best_only=True,
    period=save_period
  )

  return [logger, model_checkpoint, lrscheduler, early_stopper]
  #return [logger, model_checkpoint, lrscheduler]

def hyperparam_search(model_class, X_train, y_train, X_val, y_val,
  config=None, output=None, iters=50, save_period=1, epochs=1000,
  batch_size=128):

  if not config:
    raise ValueError('Hyperparam config has to be specified!')

  if not output:
    raise ValueError('Output has to be specified!')

  for _ in xrange(iters):

    hyperparameter_space = config.space
    hyperparams = {var: gen() for var, gen in hyperparameter_space.items()}
    hyperparam_desc = config.desc.format(**hyperparams)

    l_rate_scheduler = config.make_l_rate_scheduler(
      hyperparams['l_rate'],
      hyperparams['decay'])

    callbacks = create_callbacks(output=output, 
      prediction_layer='quat_norm', 
      run_identifier=hyperparam_desc,
      l_rate_scheduler=l_rate_scheduler,
      save_period=save_period
    )

    model = model_class(**hyperparams).model
    model.summary()

    model.fit(X_train, y_train, 
      batch_size=batch_size,
      validation_data=(X_val, y_val),
      epochs=epochs,
      callbacks=callbacks,
      verbose=True
    )

def main():
  parser = argparse.ArgumentParser()
  
  parser.add_argument('-tl', '--train-labels', nargs='+', required=True, 
    help='Path to a directory with training labels')
  parser.add_argument('-tf', '--train-features', nargs='+', required=True,
    help='Path to a numpy array with training features')

  parser.add_argument('-vl', '--val-labels', nargs='+', required=True, 
    help='Path to a directory with validation labels')
  parser.add_argument('-vf', '--val-features', nargs='+', required=True,
    help='Path to a numpy array with validation features')

  parser.add_argument('-o', '--output', required=True, 
    help='Path to an output dir with tensorboard logs, csv, checkpoints, etc')
 

  parser.add_argument('-m', '--mode', default='initial',
    choices=pose_model.PoseModel.MODES,
    help='Training mode, initial or finetuning')
  
  parser.add_argument('--top-model-weights',
    help='Top-model\'s weights for finetuning')
  parser.add_argument('--finetuning-model-arch', choices=['googlenet', 'inception_resnet_v2'],
    help='Model architecture for finetuning')
  parser.add_argument('--finetuning-model-dataset', choices=['places365', 'imagenet'],
    help='Dataset on which finetuning model was pretrained ')

  parser.add_argument('-tm', '--top-model-type', default='regressor',
    choices=pose_model.TOPMODELS.keys(),
    help='Top model to use for regression')
  parser.add_argument('--seq-len', type=int,
    help='If top-model-type is set to \'lstm\', then seq-len has to be specified!')

  parser.add_argument('--loss', default='naive_weighted',
    choices=losses.LOSSES.keys(),
    help='Loss function to use for optimization')
  parser.add_argument('-hp', '--hyperparam-config', required=True,
    help='Python file with hyperparameter configuration')

  parser.add_argument('-i', '--iters', type=int, default=50,
    help='Number of iterations for the random hyperparameter search')
  parser.add_argument('-e', '--epochs', type=int, default=1000,
    help='Number of epochs per iteration')
  parser.add_argument('-bs', '--batch-size', type=int, default=128,
    help='Batch size')
  parser.add_argument('-s', '--save-period', type=int, default=1,
    help='Number of epochs to save a checkpoint')

  #parser.add_argument('-c', '--checkpoint', type=str)
  
  args = parser.parse_args()

  hyperparam_config = importlib.import_module(args.hyperparam_config)
  


  train_features_arr = [np.squeeze(np.load(f)) for f in args.train_features]
  train_labels_arr   = [load_labels(l) for l in args.train_labels]

  val_features_arr = [np.squeeze(np.load(f)) for f in args.val_features]
  val_labels_arr   = [load_labels(l) for l in args.val_labels]

  if args.top_model_type == 'lstm':
    '''Make sequences here'''
    if not args.seq_len:
      raise ValueError('Sequence length has to be defined in LSTM mode!')


    # train_features, train_labels = [], []
    # val_features, val_labels = [], []

    # for train_f, train_l in zip(train_features_arr, train_labels_arr):
    #   tf_seq, tl_seq = make_sequences(train_f, train_l, args.seq_len)
    #   train_features.append(tf_seq)
    #   train_labels.append(tl_seq)

    # for val_f, val_l in zip(val_features_arr, val_labels_arr):
    #   vf_seq, vl_seq = make_sequences(val_f, val_l, args.seq_len)
    #   val_features.append(vf_seq)
    #   val_labels.append(vl_seq)
 
  train_features = np.concatenate(train_features)
  train_labels = np.concatenate(train_labels)
  #train_labels = train_labels[..., [0,1,2,4,5,6,3]]

  val_features = np.concatenate(val_features)
  val_labels   = np.concatenate(val_labels)
  #val_labels = val_labels[..., [0,1,2,4,5,6,3]]

  print train_features.shape, train_labels.shape
  print val_features.shape, val_labels.shape
  return                                                                                                                                                                                                                                                                                                                                                                                  


  if args.mode == 'initial': 
    input_shape = train_features.shape[1]
    def model_class(**hyperparams):
      return pose_model.PoseModel(
        input_shape=(input_shape,),
        top_model_type=args.top_model_type,
        model_loss=args.loss,
        mode=args.mode,
        **hyperparams)

  elif args.mode == 'finetune':

    finetuning_model_arch = args.finetuning_model_arch
    finetuning_model_dataset = args.finetuning_model_dataset

    finetuning_model_class = FINETUNING_MODELS[finetuning_model_arch]
    finetuning_model = finetuning_model_class(
      dataset=finetuning_model_dataset, mode='finetune').model

    def model_class(**hyperparams):
      return pose_model.PoseModel(
        input_shape=None,
        top_model_type=args.top_model_type,
        model_loss=args.loss,
        mode=args.mode,
        finetune_model=finetuning_model,
        topmodel_weights=args.top_model_weights,
        **hyperparams)

  if 'homoscedastic' in args.loss:
    '''
    Actual homescedastic loss is implemented in the last layer
    as it requires trainable parameters. Therefore, labels are fed with
    dummy data and secondary input is designated for the actual labels.
    Rerouting of the data happens here.
    '''
    #train_features = [train_labels, train_features]
    train_features = [train_features, train_labels]
    train_labels = np.zeros(train_labels.shape)

    #val_features = [val_labels, val_features]
    val_features = [val_features, val_labels]
    val_labels = np.zeros(val_labels.shape)

  hyperparam_search(model_class, train_features, train_labels,
    val_features, val_labels,
    config=hyperparam_config,
    output=args.output, 
    iters=args.iters,
    epochs=args.epochs,
    save_period=args.save_period,
    batch_size=args.batch_size)

  '''
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
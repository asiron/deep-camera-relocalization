from __future__ import print_function

import quaternion

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import izip, tee

import time, contextlib, itertools, os, re

from numpy.lib.format import open_memmap

from keras.models import Model
from keras.callbacks import Callback, CSVLogger, TensorBoard

from cnn.image_utils import load_and_process
from models.metrics import PoseMetrics

LABEL_PATTERN = 'pos_[0-9]*.txt'
IMAGE_PATTERN = '^image_[0-9]{5}\.(png|jpg)$'

@contextlib.contextmanager
def timeit(name):
  start_time = time.time()
  yield
  elapsed_time = time.time() - start_time
  tqdm.write('[{}] finished in {} ms'.format(name, int(elapsed_time * 1000)))

def pairwise(iterable):
  a, b = tee(iterable)
  next(b, None)
  return izip(a, b)

def grouper(iterable, n, fillvalue=None):
  "Collect data into fixed-length chunks or blocks"
  args = [iter(iterable)] * n
  return itertools.izip_longest(fillvalue=fillvalue, *args)

def make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)

def find_files(directory, regex):
  files = os.listdir(directory)
  files = [f for f in files if re.search(regex, f)]
  files = sorted([os.path.join(directory, f) for f in files])
  return files

def load_labels(directory, pattern=LABEL_PATTERN):
  labels = []
  label_files = find_files(directory, pattern)
  for label_file in label_files:
    with open(label_file, 'r') as file:
      parsed_line = map(float, file.readlines()[0].split(','))
      pose = parsed_line[1:]
      labels.append(pose)
  return np.array(labels)

def generate_images_from_filenames(image_filenames, batch_size=10, 
  resize=(224, 224), random_crops=False, func=lambda x: x):
  '''
  Generator, which yields processed images in batches from a directory
  Preprocessing does the following:
    resizes image to a given dimensions with a center crop
    applies a function at the end if any is given
  '''
  if random_crops > 0:
    if batch_size % random_crops != 0:
      raise ValueError('Number of random crops has to divide batch size!')
    bs = batch_size // random_crops
    process_file_func = lambda f: func(
      load_and_process(f, resize, random_crop=True))
    process_file = lambda f: [process_file_func(f) for _ in xrange(random_crops)]
  else:
    bs = batch_size
    process_file = lambda f: func(load_and_process(f, resize))

  def generator():

    for filenames_batch in grouper(image_filenames, bs):
      batch = np.array([process_file(f) for f in filenames_batch if f != None])
      yield batch.reshape((-1,) + batch.shape[-3:])

    '''wtf, predict_generator keeps calling next() even after all the steps'''
    while True:
      yield

  steps = len(image_filenames) / float(bs)
  return generator(), int(np.ceil(steps))

def generate_images(directory, batch_size=10, resize=(299, 299), 
  func=lambda x: x, pattern=IMAGE_PATTERN, random_crops=False):
  image_filenames = find_files(directory, pattern)
  return generate_images_from_filenames(image_filenames, batch_size=batch_size,
    resize=resize, random_crops=random_crops, func=func)

def split_and_pad(sequence, seq_len):
  full_subseq_count = len(sequence) // seq_len
  last_full_subseq_ind = int(full_subseq_count * seq_len)

  full_subseqs = []
  if full_subseq_count != 0:
    full_subseqs = np.split(sequence[:last_full_subseq_ind], full_subseq_count)

  last_subseq = sequence[last_full_subseq_ind:]
  if len(last_subseq):
    padding_len = seq_len - len(last_subseq)
    padding_begin = int(np.floor(padding_len / 2.0))
    padding_end   = int(np.ceil(padding_len  / 2.0))
    padding_axes = ((padding_begin, padding_end),) + ((0,0),) * (sequence.ndim-1)
    last_subseq_padded = np.lib.pad(last_subseq, padding_axes, 'reflect')
    full_subseqs.append(last_subseq_padded)

  full_subseqs = np.stack(full_subseqs, axis=0)
  assert len(full_subseqs) is int(np.ceil(len(sequence) / float(seq_len)))

  return full_subseqs

def pad_sequences(features_arr, labels_arr, seq_len):
  total = len(features_arr)
  padded_feature_arr, padded_label_arr = [], []
  gen = itertools.izip(features_arr, labels_arr)
  for f_seq, l_seq in tqdm(gen, total=total):
    padded_feature_seqs = split_and_pad(f_seq, seq_len)
    padded_label_seqs   = split_and_pad(l_seq, seq_len)

    padded_feature_arr.append(padded_feature_seqs)
    padded_label_arr.append(padded_label_seqs)

    print(padded_feature_seqs.shape, padded_label_seqs.shape)

  padded_feature_arr = np.concatenate(padded_feature_arr, axis=0)
  padded_label_arr = np.concatenate(padded_label_arr, axis=0)

  return padded_feature_arr, padded_label_arr

def reshape_to_stateful_input(dataset, batch_size, subseq_len=1):

  print(len(dataset), batch_size, subseq_len)

  assert len(dataset) % batch_size == 0
  assert dataset.shape[1] % subseq_len == 0

  splits = len(dataset) / batch_size  
  feature_shape = dataset.shape[2:]

  dataset = dataset.reshape((len(dataset), -1, subseq_len) + feature_shape)

  reshape_dims = (-1, subseq_len) + feature_shape
  permute_axes = (1, 0) + tuple(range(2, dataset.ndim))

  transposed = [a.transpose(permute_axes) for a in np.split(dataset, splits)]
  return np.concatenate(transposed).reshape(reshape_dims)

def make_stateful_sequences(features, labels, 
                            seq_len=None, batch_size=None, subseq_len=None):
  features, labels = pad_sequences(features, labels, seq_len)\

  perm = np.random.permutation(len(features))
  features = features[perm]
  labels = labels[perm]
  print('Padded shapes', features.shape, labels.shape)
  features = np.squeeze(reshape_to_stateful_input(features, batch_size, subseq_len=subseq_len))
  labels = np.squeeze(reshape_to_stateful_input(labels, batch_size, subseq_len=subseq_len))
  return features, labels

def make_standard_sequences(Xs, Ys, 
  subseq_len=None, step=1, repeat=True):

  if repeat:
    for (X_seq, Y_seq) in izip(Xs, Ys):
      X_sequences, Y_sequences = [], []
      for n in xrange(0, len(X_seq) - subseq_len + 1, step):
        X_sequences.append(X_seq[n:n+subseq_len])
        Y_sequences.append(Y_seq[n:n+subseq_len])
      yield np.array(X_sequences), np.array(Y_sequences)    

  else:
    for (X_seq, Y_seq) in izip(Xs, Ys):
      X_sequences, Y_sequences = [], []

      block_len = subseq_len * step
      n_blocks = len(X_seq) // block_len

      for n in xrange(n_blocks):
        start_idx, end_idx = n*block_len, (n+1)*block_len
        base_idxs = np.arange(start_idx, end_idx, step)
        for i in xrange(step):
          print(base_idxs+i)
          X_sequences.append(X_seq[base_idxs+i])
          Y_sequences.append(Y_seq[base_idxs+i])

      yield np.array(X_sequences), np.array(Y_sequences)

def search_layer(model, layer_name):
  found_layer = None
  for m in (l for l in model.layers if type(l) is Model):
    l = search_layer(m, layer_name)
    if l and l.name == layer_name:
      return l

  ll = [l for l in model.layers if l.name == layer_name]
  return ll[0] if len(ll) == 1 else None 

def get_mmaps(directory, pattern):
  return find_files(directory, pattern)
  
def delete_mmaps(directory, pattern):
  mmap_files = get_mmaps(directory, pattern)
  delete = lambda f: os.remove(f)  
  map(delete, mmap_files)

def load_mmaps(directory, pattern):
  mmap_files = get_mmaps(directory, pattern)
  load = lambda f: np.load(f, mmap_mode='r')  
  return map(load, mmap_files)

def concatenate_without_loading(directory, 
  pattern=None, filename=None, func=lambda x: x):

  if pattern is None or filename is None:
    raise ValueError('Pattern and filename have to be defined!')

  mmaps = load_mmaps(directory, pattern)
  merged_path = os.path.join(directory, filename)
  concatenate_mmaps(mmaps, merged_path, func=func)
  delete_mmaps(directory, pattern)

def concatenate_mmaps(mmaps, filename, func=lambda x: x):

  mmaps = map(func, mmaps)
  total_size = reduce(lambda x,y: x + len(y), mmaps, 0)

  merged_shape = (total_size,) + mmaps[0].shape[1:]
  merged_mmap = open_memmap(filename,
    dtype=np.float32, mode='w+', shape=merged_shape)

  start_idx, end_idx = 0, 0
  for n, mmap in enumerate(mmaps):

    end_idx = start_idx + len(mmap)
    merged_mmap[start_idx:end_idx] = mmap
    start_idx = end_idx

  del merged_mmap

def get_starting_indicies(arrays):
  lens = [len(x) for x in arrays]
  return np.insert(np.cumsum(lens), 0, 0)

class ResetStatesCallback(Callback):
  
  def __init__(self, interval=None):
    if interval is None:
      raise ValueError('Interval has to be specified!')

    self.counter = 0
    self.interval = interval

  def on_batch_begin(self, batch, logs={}):
    if self.counter % self.interval == 0:
      self.model.reset_states()
      print("Reseting Model's internal states...")
    self.counter += 1

class ExtendedLogger(Callback):

  val_data_metrics = {}

  def __init__(self, prediction_layer, output_dir='./tmp',
    stateful=False, stateful_reset_interval=None, starting_indicies=None):

    if stateful and stateful_reset_interval is None:
      raise ValueError('If model is stateful, then seq-len has to be defined!')

    super(ExtendedLogger, self).__init__()
    
    self.csv_dir  = os.path.join(output_dir, 'csv')
    self.tb_dir   = os.path.join(output_dir, 'tensorboard')
    self.pred_dir = os.path.join(output_dir, 'predictions')
    self.plot_dir = os.path.join(output_dir, 'plots')

    make_dir(self.csv_dir)
    make_dir(self.tb_dir)
    make_dir(self.plot_dir)
    make_dir(self.pred_dir)

    self.stateful = stateful
    self.stateful_reset_interval = stateful_reset_interval
    self.starting_indicies = starting_indicies
    self.csv_logger = CSVLogger(os.path.join(self.csv_dir, 'run.csv'))
    self.tensorboard = TensorBoard(log_dir=self.tb_dir, write_graph=True) 
    self.prediction_layer = prediction_layer

  def set_params(self, params):
    super(ExtendedLogger, self).set_params(params)
    self.tensorboard.set_params(params)
    self.tensorboard.batch_size = params['batch_size']
    self.csv_logger.set_params(params)

  def set_model(self, model):
    super(ExtendedLogger, self).set_model(model)
    self.tensorboard.set_model(model)
    self.csv_logger.set_model(model)

  def on_batch_begin(self, batch, logs=None):
    self.csv_logger.on_batch_begin(batch, logs=logs)
    self.tensorboard.on_batch_begin(batch, logs=logs)

  def on_batch_end(self, batch, logs=None):
    self.csv_logger.on_batch_end(batch, logs=logs)
    self.tensorboard.on_batch_end(batch, logs=logs)

  def on_train_begin(self, logs=None):
    self.csv_logger.on_train_begin(logs=logs)
    self.tensorboard.on_train_begin(logs=logs)

  def on_train_end(self, logs=None):
    self.csv_logger.on_train_end(logs=logs)
    self.tensorboard.on_train_end(logs)

  def on_epoch_begin(self, epoch, logs=None):
    self.csv_logger.on_epoch_begin(epoch, logs=logs)
    self.tensorboard.on_epoch_begin(epoch, logs=logs)

  def on_epoch_end(self, epoch, logs=None):

    with timeit('metrics'):
      
      outputs = self.model.get_layer(self.prediction_layer).output
      self.prediction_model = Model(inputs=self.model.input, outputs=outputs)

      batch_size = self.params['batch_size']

      if isinstance(self.validation_data[-1], float):
        val_data = self.validation_data[:-2]
      else:
        val_data = self.validation_data[:-1]

      y_true = val_data[1]

      callback = None
      if self.stateful:
        callback = ResetStatesCallback(interval=self.stateful_reset_interval)
        callback.model = self.prediction_model

      y_pred = self.prediction_model.predict(val_data[:-1], 
        batch_size=batch_size, verbose=1, callback=callback)

      print(y_true.shape, y_pred.shape)

      self.write_prediction(epoch, y_true, y_pred)

      y_true = y_true.reshape((-1, 7))
      y_pred = y_pred.reshape((-1, 7))

      self.save_error_histograms(epoch, y_true, y_pred)
      self.save_topview_trajectories(epoch, y_true, y_pred)

      new_logs = {name: np.array(metric(y_true, y_pred))
        for name, metric in self.val_data_metrics.items()}
      logs.update(new_logs)

      homo_logs = self.try_add_homoscedastic_params()
      logs.update(homo_logs)

      self.tensorboard.validation_data = self.validation_data
      self.csv_logger.validation_data = self.validation_data

      self.tensorboard.on_epoch_end(epoch, logs=logs)
      self.csv_logger.on_epoch_end(epoch, logs=logs)

  def add_validation_metrics(self, metrics_dict):
    self.val_data_metrics.update(metrics_dict)

  def add_validation_metric(self, name, metric):
    self.val_data_metrics[name] = metric

  def try_add_homoscedastic_params(self):
    homo_pos_loss_layer = search_layer(self.model, 'homo_pos_loss')
    homo_quat_loss_layer = search_layer(self.model, 'homo_quat_loss')

    if homo_pos_loss_layer:
      homo_pos_log_vars = np.array(homo_pos_loss_layer.get_weights()[0])
      homo_quat_log_vars = np.array(homo_quat_loss_layer.get_weights()[0])
      return {
        'pos_log_var' : np.array(homo_pos_log_vars), 
        'quat_log_var' : np.array(homo_quat_log_vars), 
      }
    else:
      return {}

  def write_prediction(self, epoch, y_true, y_pred):
    filename = '{:04d}_predictions.npy'.format(epoch)
    filename = os.path.join(self.pred_dir, filename)
    arr = {'y_pred': y_pred, 'y_true': y_true}
    np.save(filename, arr)

  def save_topview_trajectories(self, epoch, y_true, y_pred, max_segment=1000):
   
    if self.starting_indicies is None:
      self.starting_indicies = {'valid' : range(0, 4000, 1000) + [4000]}

    for begin, end in pairwise(self.starting_indicies['valid']):

      diff = end - begin
      if diff > max_segment:
        subindicies = range(begin, end, max_segment) + [end]
        for b, e in pairwise(subindicies):
          self.save_trajectory(epoch, y_true, y_pred, b, e)

      self.save_trajectory(epoch, y_true, y_pred, begin, end)


  def save_trajectory(self, epoch, y_true, y_pred, begin, end):
    true_xy, pred_xy = y_true[begin:end, :2], y_pred[begin:end, :2]
    
    true_q = quaternion.as_quat_array(y_true[begin:end, [6,3,4,5]])
    true_q = quaternion.as_euler_angles(true_q)[1]

    pred_q = quaternion.as_quat_array(y_pred[begin:end, [6,3,4,5]])
    pred_q = quaternion.as_euler_angles(pred_q)[1]

    plt.clf()

    plt.plot(true_xy[:, 0], true_xy[:, 1], 'g-')
    plt.plot(pred_xy[:, 0], pred_xy[:, 1], 'r-')

    for ((x1, y1), (x2, y2)) in zip(true_xy, pred_xy):
      plt.plot([x1, x2], [y1, y2], 
               color='k', linestyle='-', 
               linewidth=0.3, alpha=0.2)


    plt.grid(True)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Top-down view of trajectory')
    plt.axis('equal')

    x_range = (np.min(true_xy[:, 0])-.2, np.max(true_xy[:, 0])+.2)
    y_range = (np.min(true_xy[:, 1])-.2, np.max(true_xy[:, 1])+.2)

    plt.xlim(x_range)
    plt.ylim(y_range)

    filename = 'epoch={epoch:04d}_begin={begin:04d}_end={end:04d}_trajectory.pdf' \
      .format(epoch=epoch, begin=begin, end=end)
    filename = os.path.join(self.plot_dir, filename)
    plt.savefig(filename)

  def save_error_histograms(self, epoch, y_true, y_pred):
    pos_errors = PoseMetrics.abs_errors_position(y_true, y_pred)
    pos_errors = np.sort(pos_errors)

    angle_errors = PoseMetrics.abs_errors_orienation(y_true, y_pred)
    angle_errors = np.sort(angle_errors)

    size = len(y_true)
    ys = np.arange(size)/float(size)
    
    plt.clf()

    plt.subplot(2, 1, 1)
    plt.title('Empirical CDF of absolute errors')
    plt.grid(True)
    plt.plot(pos_errors, ys, 'k-')
    plt.xlabel('Absolute Position Error (m)')
    plt.xlim(0, 1.2)

    plt.subplot(2, 1, 2)
    plt.grid(True)
    plt.plot(angle_errors, ys, 'r-')
    plt.xlabel('Absolute Angle Error (deg)')
    plt.xlim(0, 70)

    filename = '{:04d}_cdf.pdf'.format(epoch)
    filename = os.path.join(self.plot_dir, filename)
    plt.savefig(filename)
import numpy as np
import time, contextlib, itertools, os, re

from tqdm import tqdm

from keras.models import Model
from keras.callbacks import Callback, CSVLogger, TensorBoard

from cnn.image_utils import load_and_process

LABEL_PATTERN = 'pos_[0-9]*.txt'
IMAGE_PATTERN = '^image_[0-9]{5}\.(png|jpg)$'

@contextlib.contextmanager
def timeit(name):
  start_time = time.time()
  yield
  elapsed_time = time.time() - start_time
  tqdm.write('[{}] finished in {} ms'.format(name, int(elapsed_time * 1000)))

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
  resize=(299,299), func=lambda x: x):

  def generator():
    process_file = lambda f: func(load_and_process(f, resize))

    for filenames_batch in grouper(image_filenames, batch_size):
      batch = [process_file(f) for f in filenames_batch if f != None]
      yield np.array(batch)

    '''wtf, predict_generator keeps calling next() even after all the steps'''
    while True:
      yield

  steps = len(image_filenames) / float(batch_size)
  return generator(), int(np.ceil(steps))

def generate_images(directory, batch_size=10, 
  resize=(299, 299), func=lambda x: x, pattern=IMAGE_PATTERN):
  '''
  Generator, which yields processed images in batches from a directory
  Preprocesing does the following:
    resizes image to a given dimensions with a center crop
    scales image s.t each pixel is in [-1, 1] range
    applies a function at the end if any is given
  '''
  image_filenames = find_files(directory, pattern)
  return generate_images_from_filenames(image_filenames, batch_size=batch_size,
    resize=resize, func=func)

def make_sequences(Xs, Ys, seqlen, step = 1):
  Xseq, Yseq = [], []
  for i in xrange(0, Xs.shape[0]-seqlen+1, step):
    Xseq.append(Xs[i: i+seqlen])
    Yseq.append(Ys[i: i+seqlen])
  return np.array(Xseq), np.array(Yseq)

def search_layer(model, layer_name):
  found_layer = None
  for m in (l for l in model.layers if type(l) is Model):
    l = search_layer(m, layer_name)
    if l and l.name == layer_name:
      return l

  ll = [l for l in model.layers if l.name == layer_name]
  return ll[0] if len(ll) == 1 else None 

class ExtendedLogger(Callback):

  val_data_metrics = {}

  def __init__(self, prediction_layer, csv_dir='./csv', tb_dir='./tensorboard'):
    super(ExtendedLogger, self).__init__()
    make_dir(csv_dir)
    make_dir(tb_dir)
    self.csv_logger = CSVLogger(os.path.join(csv_dir, 'run.csv'))
    self.tensorboard = TensorBoard(log_dir=tb_dir, 
      write_graph=True, 
      #write_grads=True,
      histogram_freq=30)

    self.prediction_layer = prediction_layer

  def set_params(self, params):
    super(ExtendedLogger, self).set_params(params)
    self.tensorboard.set_params(params)
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

      val_data = self.validation_data[:-2]
      y_true = val_data[1]
      y_pred = self.prediction_model.predict(val_data[:-1], 
        batch_size=batch_size, verbose=1)

      new_logs = {name: np.array(metric(y_true, y_pred))
        for name, metric in self.val_data_metrics.items()}
      logs.update(new_logs)

      self.tensorboard.validation_data = self.validation_data
      self.csv_logger.validation_data = self.validation_data

      self.csv_logger.on_epoch_end(epoch, logs=logs)
      self.tensorboard.on_epoch_end(epoch, logs=logs)

  def add_validation_metrics(self, metrics_dict):
    self.val_data_metrics.update(metrics_dict)

  def add_validation_metric(self, name, metric):
    self.val_data_metrics[name] = metric

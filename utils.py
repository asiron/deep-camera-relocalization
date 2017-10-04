import numpy as np
import time, contextlib, itertools, os, re

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
  print('[{}] finished in {} ms'.format(name, int(elapsed_time * 1000)))

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

  print 'Labels loaded!'
  return np.array(labels)

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

class ExtendedLogger(Callback):

  val_data_metrics = {}

  def __init__(self, prediction_layer, csv_dir='./csv', tb_dir='./tensorboard'):
    super(ExtendedLogger, self).__init__()
    make_dir(csv_dir)
    make_dir(tb_dir)
    self.prediction_layer = prediction_layer
    self.csv_logger = CSVLogger(os.path.join(csv_dir, 'run.csv'))
    self.tensorboard = TensorBoard(log_dir=tb_dir, write_graph=True)

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

    batch_size = self.params['batch_size']
    prediction_model = Model(inputs=self.model.input, 
      outputs=self.model.get_layer(self.prediction_layer).output)

    y_true = self.validation_data[1]
    with timeit('prediction'):
      y_pred = prediction_model.predict(
        self.validation_data[0], batch_size=batch_size, verbose=1)

    with timeit('metrics'):
      new_logs = {name: np.array(metric(y_true, y_pred))
        for name, metric in self.val_data_metrics.items()}
      logs.update(new_logs)

    self.csv_logger.on_epoch_end(epoch, logs=logs)
    self.tensorboard.on_epoch_end(epoch, logs=logs)

  def add_validation_metrics(self, metrics_dict):
    self.val_data_metrics.update(metrics_dict)

  def add_validation_metric(self, name, metric):
    self.val_data_metrics[name] = metric

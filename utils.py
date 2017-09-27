import os, re

import numpy as np

from keras.models import Model
from keras.callbacks import TensorBoard, CSVLogger

def find_files(directory, regex):
  files = os.listdir(directory)
  files = [f for f in files if re.search(regex, f)]
  files = sorted([os.path.join(directory, f) for f in files])
  return files

class ImprovedTensorBoardWithCSVLogger(CSVLogger, TensorBoard):

  val_data_metrics = {}

  def __init__(self, prediction_layer, *args, **kwargs):
    super(ImprovedTensorBoard, self).__init__(*args, **kwargs)
    self.prediction_layer = prediction_layer

  def on_epoch_end(self, epoch, logs=None):

    batch_size = self.params['batch_size']

    prediction_model = Model(inputs=self.model.input, 
      outputs=self.model.get_layer(self.prediction_layer).output)

    y_true = self.validation_data[1]
    y_pred = prediction_model.predict(self.validation_data[0], 
      batch_size=batch_size, verbose=1)

    new_logs = {name: np.array(metric(y_true, y_pred))
      for name, metric in self.val_data_metrics}

    logs.update(new_logs)
    super(ImprovedTensorBoard, self).on_epoch_end(self, epoch, logs=logs)


  def add_validation_metric(name, metric):
    self.val_data_metrics[name] = metric

# for name, metric in validation_data_metrics.items():
#   summary = tf.Summary()  
#   summary_value = summary.value.add()
#   summary_value.tag = name
#   summary_value.simple_value = metric(y_true, y_pred)
#   self.writer.add_summary(summary, epoch)

# self.writer.flush()
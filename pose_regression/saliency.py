from __future__ import print_function

import numpy as np
import argparse, os, importlib, itertools

from sklearn.model_selection import train_test_split
from keras.models import load_model, Sequential, Model
from keras.callbacks import (
  LearningRateScheduler, 
  ModelCheckpoint,
  EarlyStopping,
  ReduceLROnPlateau
)

from models import pose_model, losses, metrics, layers
from utils import (
  generate_images_from_filenames, make_dir, search_layer)
import cnn

import keras.backend as K


def main():
  parser = argparse.ArgumentParser()
  
  parser.add_argument('-i', '--images', nargs='+', required=True, 
    help='Paths to images to be visualzied with saliency')

  parser.add_argument('-o', '--output', required=True, 
    help='Path to an output dir where the saliency maps will be saved')
 
  parser.add_argument('-m', '--mode', default='vanilla',
    choices=['vanilla', 'guided'],
    help='saliency Map computation mode')
  
  parser.add_argument('--smooth', action='store_true', dest='smooth',
    help='Smooth the gradient')
  parser.add_argument('--no-smooth', action='store_false', dest='smooth',
    help='Do not smooth the gradient')
  parser.set_defaults(smooth=True)

  parser.add_argument('--model-weights',
    help='Model\'s weights to be used for saliency Map computation')

  parser.add_argument('-bs', '--batch-size', type=int, default=32,
    help='Batch size')

  args = parser.parse_args()

  params = args.model_weights.split('/')

  top_model_type = params[-4].split('_')[0]
  model_loss = params[-4].split('_')[1]
  finetuning_model_arch = params[-4].split('_')[2]
  finetuning_model_dataset = params[-4].split('_')[3]
  seq_len = int(params[-4].split('_')[4].split('=')[1])

  hyperparams = dict(map(lambda x: tuple(x.split('=')), params[-3].split(',')))

  print(hyperparams)


  custom_objects = {
    'QuaternionNormalization' : layers.QuaternionNormalization,
    'QuaternionErrorPoseLoss' : losses.QuaternionErrorPoseLoss,
    'HomoscedasticLoss' : layers.HomoscedasticLoss,
    '_dummy_loss' : losses.DummyLoss(),
    'quaternion_err_weighted_pose_loss' : losses.QuaternionErrorWeightedPoseLoss(gamma=1, beta=463.7)
  }


  from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Reshape, Dense, Dropout, Lambda, CuDNNLSTM, Concatenate

  input = Input(shape=(224, 224, 3), name='cnn_input')
  
  conv1_1 = Conv2D(64, (3,3), strides=(1,1), activation='relu', name='conv1_1', padding='same')(input)
  conv1_2 = Conv2D(64, (3,3), strides=(1,1), activation='relu', name='conv1_2', padding='same')(conv1_1)
  pool1   = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='pool1', padding='same')(conv1_2)

  conv2_1 = Conv2D(128, (3,3), strides=(1,1), activation='relu', name='conv2_1', padding='same')(pool1)
  conv2_2 = Conv2D(128, (3,3), strides=(1,1), activation='relu', name='conv2_2', padding='same')(conv2_1)
  pool2   = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='pool2', padding='same')(conv2_2)

  conv3_1 = Conv2D(256, (3,3), strides=(1,1), activation='relu', name='conv3_1', padding='same')(pool2)
  conv3_2 = Conv2D(256, (3,3), strides=(1,1), activation='relu', name='conv3_2', padding='same')(conv3_1)
  conv3_3 = Conv2D(256, (3,3), strides=(1,1), activation='relu', name='conv3_3', padding='same')(conv3_2)
  pool3   = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='pool3', padding='same')(conv3_3)

  conv4_1 = Conv2D(512, (3,3), strides=(1,1), activation='relu', name='conv4_1', padding='same')(pool3)
  conv4_2 = Conv2D(512, (3,3), strides=(1,1), activation='relu', name='conv4_2', padding='same')(conv4_1)
  conv4_3 = Conv2D(512, (3,3), strides=(1,1), activation='relu', name='conv4_3', padding='same')(conv4_2)
  pool4   = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='pool4', padding='same')(conv4_3)

  conv5_1 = Conv2D(512, (3,3), strides=(1,1), activation='relu', name='conv5_1', padding='same')(pool4)
  conv5_2 = Conv2D(512, (3,3), strides=(1,1), activation='relu', name='conv5_2', padding='same')(conv5_1)
  conv5_3 = Conv2D(512, (3,3), strides=(1,1), activation='relu', name='conv5_3', padding='same')(conv5_2)
  pool5   = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='pool5', padding='same')(conv5_3)
  
  gap2d = GlobalAveragePooling2D()(pool5)

  dense_1 = Dense(1024, activation='relu')(gap2d)

  dropout_1 = Dropout(0.0001)(dense_1)

  reshaped = Reshape((32, 32))(dropout_1)
  reshaped_reversed = Lambda(lambda x: K.reverse(x, axes=1))(reshaped)

  transposed = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(reshaped)
  transposed_reversed = Lambda(lambda x: K.reverse(x, axes=1))(transposed)

  lstm_left_right = CuDNNLSTM(32, return_sequences=False)(reshaped)
  lstm_right_left = CuDNNLSTM(32, return_sequences=False)(reshaped_reversed)

  lstm_top_down = CuDNNLSTM(32, return_sequences=False)(transposed)
  lstm_bottom_up = CuDNNLSTM(32, return_sequences=False)(transposed_reversed)

  merged = Concatenate()([
    lstm_left_right, lstm_right_left, lstm_top_down, lstm_bottom_up])

  dense_2   = Dense(7)(merged)
  quat_norm = layers.QuaternionNormalization(name='quat_norm')(dense_2)
  prediction = Lambda(lambda x: x, name='prediction')(quat_norm)

  model = Model(inputs=input, outputs=prediction)
  model.summary()

  model.load_weights(args.model_weights, by_name=True)
  model.load_weights('pose_regression/cnn/vgg16/hybrid1365/hybrid1365_base.h5', by_name=True)

  # top_model = load_model(args.model_weights, custom_objects=custom_objects)
  # top_model_reduced = Model(inputs=top_model.input, outputs=top_model.get_layer('prediction').output)
  # top_model_reduced.name = 'top_model'

  # base_model = cnn.vgg16.vgg16.VGG16(mode='base', dataset='hybrid1365').build()
  # base_model.name = 'cnn_extractor'

  # base_model.summary()
  # top_model_reduced.summary()

  # model = Sequential()
  # model.add(base_model)
  # model.add(top_model_reduced)

  # model.summary()

  # trained_model = pose_model.PoseModel(
  #   input_shape=None,
  #   top_model_type=top_model_type,
  #   model_loss=model_loss,
  #   mode='predict',
  #   finetuning_model_arch=finetuning_model_arch,
  #   finetuning_model_dataset=finetuning_model_dataset,
  #   topmodel_weights=args.model_weights,
  #   seq_len=seq_len,
  #   **hyperparams).build()

  # trained_model.summary()


  from vis.visualization import visualize_saliency_with_losses, overlay
  from vis.utils import utils as visutils
  from keras import activations
  import matplotlib.pyplot as plt

  from vis.backprop_modifiers import get
  from vis.losses import ActivationMaximization


  #layer_idx = visutils.find_layer_idx(model, 'prediction')

  images = '/media/labuser/Storage/arg-00/datasets/7scenes/office/test/seq-06/frame-000732.color.png'
  images, _ = generate_images_from_filenames([images], batch_size=1)
  images = itertools.takewhile(lambda b: b is not None, images)


  for modifier in ['guided', 'relu']:
    plt.figure()
    f, ax = plt.subplots(1, 2)
    plt.suptitle(modifier)

    if modifier is not None:
      modifier_fn = get(modifier)
      model = modifier_fn(model, custom_objects=custom_objects)
      #model.compile(optimizer='adam', loss='mse')

    layer = model.layers[-1]
    print(layer)
    #layer = search_layer(model, 'prediction')
    loss = [
        (ActivationMaximization(layer, None), -1)
    ]

    import tensorflow as tf

    sess = K.get_session()
    writer = tf.summary.FileWriter('logs', sess.graph)
    #print sess.run(golden_ratio)
    writer.close()


    for i, img in enumerate(images):
        grads = visualize_saliency_with_losses(model.inputs[0], loss, img, grad_modifier='absolute')
        # Lets overlay the heatmap onto original image.    
        ax[i].imshow(grads, cmap='jet')

    plt.show()

  # model.predict(X_train, y_train, 
  #   batch_size=batch_size,
  #   validation_data=(X_val, y_val),
  #   epochs=epochs,
  #   callbacks=callbacks,
  #   verbose=True,
  #   shuffle=(not stateful)
  # )


  # if 'homoscedastic' in args.loss:
  #   '''
  #   Actual homescedastic loss is implemented in the last layer
  #   as it requires trainable parameters. Therefore, labels are fed with
  #   dummy data and secondary input is designated for the actual labels.
  #   Rerouting of the data happens here.
  #   '''
  #   train_features = {'main_input' : train_features, 'labels_input' : train_labels}
  #   train_labels = np.zeros((train_labels.shape[0],))

  #   val_features = {'main_input' : val_features, 'labels_input' : val_labels}
  #   val_labels = np.zeros((val_labels.shape[0],))

  # stateful = 'stateful' in args.top_model_type



  # hyperparam_search(model_class, train_features, train_labels,
  #   val_features, val_labels,
  #   config=hyperparam_config,
  #   output=args.output, 
  #   iters=args.iters,
  #   epochs=args.epochs,
  #   save_period=args.save_period,
  #   batch_size=args.batch_size,
  #   stateful=stateful,
  #   seq_len=args.seq_len)

if __name__ == '__main__':
  main()

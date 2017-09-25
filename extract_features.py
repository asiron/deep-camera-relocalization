#!/usr/bin/env

import argparse, os, scipy, re, itertools
import numpy as np

from tqdm import tqdm

from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator

import keras.backend as K
import tensorflow as tf

from utils import find_files

POSSIBLE_MODELS = [
  'vgg16', 
  'resnet50', 
  'inception_v3', 
  'inception_resnet_v2'
]

def grouper(iterable, n, fillvalue=None):
  "Collect data into fixed-length chunks or blocks"
  args = [iter(iterable)] * n
  return itertools.izip_longest(fillvalue=fillvalue, *args)

def generate_images(directory, batch_size=10, resize=(299, 299), data_format='channel_first'):
  
  pattern = '^image_[0-9]{5}\.(png|jpg)$'
  image_filenames = find_files(directory, pattern)

  def generator():
    for filenames_batch in grouper(image_filenames, batch_size):
      #print filenames_batch
      batch = [load_and_process(f, data_format, resize) for f in filenames_batch if f != None]
      yield np.array(batch)

    '''wtf, predict_generator keeps calling next() even after all the steps'''
    while True:
      yield

  steps = len(image_filenames) / float(batch_size)
  print int(np.ceil(steps))
  return generator(), int(np.ceil(steps))

def load_and_process(image_filename, data_format, resize):
  image = scipy.misc.imread(image_filename, mode='RGB')
  return preprocess_image(image, data_format, *resize)

def preprocess_image(img, data_format, new_height, new_width):
  if img.shape[0] > img.shape[1]:
    raise ValueError('Image is not in HW format (Height, Width')

  resized_height = new_height
  resized_width = int(new_width * (img.shape[1] / float(img.shape[0])))
  
  img = scipy.misc.imresize(img, (resized_height, resized_width))
  img = center_crop(img, new_height, new_width)
  img = scale_img(img)
  img = np.transpose(img, (2, 0, 1)) if data_format == 'channels_first' else img
  return img 

def center_crop(img, new_height, new_width):
  y, x = img.shape[:2]
  startx = x // 2 - (new_width  // 2)
  starty = y // 2 - (new_height // 2)    
  return img[starty:starty+new_height, startx:startx+new_width]

def scale_img(img):
  img = img.astype(float)
  img /= 255.0
  img -= 0.5
  img *= 2
  return img

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('dataset', help='Path to the dataset')
  parser.add_argument('output', help='Path to an output directory')
  #parser.add_argument('-i', '--img_size', required=True, type=img_size_type,
  #  help='Size of images in the dataset in HxW format, e.g. 720x1280')
  parser.add_argument('-c', '--channels', default='first',
    choices=['first', 'last'],
    help='Tensor ordering: NHWC or NCHW')
  parser.add_argument('-m', '--model', default='inception_resnet_v2', 
    choices=POSSIBLE_MODELS,
    help='Pretrained CNN model from which we extract features')
  
  args = parser.parse_args()

  '''
  config = tf.ConfigProto()
  config.gpu_options.allow_growth=True
  sess = tf.Session(config=config)
  K.set_session(sess)
  '''
  
  output_dir = args.output
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  batch_size = 16
  resize = (299, 299)

  if args.channels == 'first':
    K.set_image_data_format('channels_first')
    input_shape = (3,) + resize
  elif args.channels == 'last':
    input_shape = resize + (3,)

  base_model = InceptionV3(weights='imagenet', include_top=True, input_shape=input_shape)
  model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
 

  '''
  steps = int(np.ceil(dataset_img_count / float(batch_size)))
  original_img_size = args.img_size
  dataset_img_count = len(os.listdir(os.path.join(args.dataset, 'images')))

  image_datagen = ImageDataGenerator(
    preprocessing_function=lambda i: preprocess_image(i, *resize, data_format=args.channels)
  )
  image_generator = image_datagen.flow_from_directory(
    args.dataset,
    class_mode=None,
    shuffle=False,
    target_size=original_img_size,
    batch_size=batch_size,
    save_to_dir=output_dir
  )
  '''
  image_generator, steps = generate_images(args.dataset, 
    batch_size=batch_size, 
    resize=resize, 
    data_format=K.image_data_format()
  )

  #for i in image_generator:
  # print i.shape

  features = model.predict_generator(image_generator, steps, verbose=True)
  print features.shape

  output_features = os.path.join(output_dir, 'features.npy')
  np.save(output_features, features)
  
if __name__ == '__main__':
  main()
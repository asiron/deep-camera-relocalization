#!/usr/bin/env

import argparse, os
import numpy as np
from tqdm import tqdm

from cnn.googlenet import GoogleNet
from cnn.inception_resnet_v2 import InceptionResNetV2
from cnn.image_utils import load_and_process

from utils import find_files, grouper, make_dir

MODELS = {
  'googlenet' : GoogleNet,
  'inception_resnet_v2' : InceptionResNetV2
}

DATASETS = [
  'places365',
  'imagenet'
]

def generate_images(directory, batch_size=10, resize=(299, 299), func=lambda x: x):
  '''
  Generator, which yields processed images in batches from a directory
  Preprocesing does the following:
    resizes image to a given dimensions with a center crop
    scales image s.t each pixel is in [-1, 1] range
    applies a function at the end if any is given
  '''
  pattern = '^image_[0-9]{5}\.(png|jpg)$'
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

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('images', 
    help='Path to the images directory that contains images')
  parser.add_argument('output', help='Path to an output directory')
  parser.add_argument('-m', '--model', default='inception_resnet_v2', 
    choices=MODELS.keys(),
    help='Pretrained CNN model from which we extract features')
  parser.add_argument('-d', '--dataset', default='places365', 
    choices=DATASETS,
    help='Dataset on which the CNN model was pretrained')
  
  args = parser.parse_args()

  model = MODELS[args.model](dataset=args.dataset, mode='extract')
  
  input_shape = model.input_shape[:2]

  image_generator, steps = generate_images(
    args.images, 
    batch_size=10, 
    resize=input_shape,
    func=model.preprocess_image
  )

  cnn_f, finetune_f = model.model.predict_generator(image_generator, steps, verbose=True)

  output_dir = args.output
  make_dir(output_dir)

  finetune_features = os.path.join(output_dir, 'finetune_features.npy')
  cnn_features = os.path.join(output_dir, 'cnn_features.npy')

  np.save(finetune_features, np.array(finetune_f[0]))
  np.save(cnn_features, np.array(cnn_f[0]))
  
  
if __name__ == '__main__':
  main()
#!/usr/bin/env

import argparse, os
import numpy as np
from tqdm import tqdm

from cnn.googlenet import GoogleNet
from cnn.inception_resnet_v2 import InceptionResNetV2

from utils import make_dir, generate_images, IMAGE_PATTERN

MODELS = {
  'googlenet' : GoogleNet,
  'inception_resnet_v2' : InceptionResNetV2
}

DATASETS = [
  'places365',
  'imagenet'
]

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
  parser.add_argument('-p', '--pattern', 
    help='Image grep pattern for looking in images directory')
  
  args = parser.parse_args()

  model = MODELS[args.model](dataset=args.dataset, mode='extract')
  
  input_shape = model.input_shape[:2]

  image_pattern = args.pattern or IMAGE_PATTERN
  image_generator, steps = generate_images(
    args.images, 
    batch_size=10, 
    resize=input_shape,
    func=model.preprocess_image,
    pattern=image_pattern
  )

  cnn_f, finetune_f = model.model.predict_generator(image_generator, steps, verbose=True)

  output_dir = args.output
  make_dir(output_dir)

  finetune_features = os.path.join(output_dir, 'finetune_features.npy')
  cnn_features = os.path.join(output_dir, 'cnn_features.npy')

  np.save(finetune_features, np.array(finetune_f))
  np.save(cnn_features, np.array(cnn_f))
  
  
if __name__ == '__main__':
  main()
import numpy as np
import argparse, os

from tqdm import tqdm
from itertools import izip, takewhile, islice

from ..cnn.googlenet import GoogleNet
from ..cnn.inception_resnet_v2 import InceptionResNetV2

from ..utils import (
  make_dir, generate_images, concatenate_without_loading,
  IMAGE_PATTERN)

MODELS = {
  'googlenet' : GoogleNet,
  'inception_resnet_v2' : InceptionResNetV2
}

DATASETS = [
  'places205',
  'places365',
  'imagenet'
]

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('images', 
    help='Path to the images directory that contains images')
  parser.add_argument('output', help='Path to an output directory')
  parser.add_argument('--meanfile',
    help='Path to the numpy array containing meanfile')
  parser.add_argument('-m', '--model', default='inception_resnet_v2', 
    choices=MODELS.keys(),
    help='Pretrained CNN model from which we extract features')
  parser.add_argument('-d', '--dataset', default='places365', 
    choices=DATASETS,
    help='Dataset on which the CNN model was pretrained')
  parser.add_argument('-p', '--pattern', 
    help='Image grep pattern for looking in images directory')
  parser.add_argument('-b', '--batch-size', type=int, default=10,
    help='Batch size for image feature extraction')
  parser.add_argument('-r', '--random-crops', type=int, default=0,
    help='Random crops per image')
  parser.add_argument('-s', '--seed', type=int, default=42,
    help='PRNG seed')

  args = parser.parse_args()

  np.random.seed(args.seed)

  output_dir = args.output
  make_dir(output_dir)

  kwargs = {k: vars(args)[k] for k in ('dataset', 'meanfile')}

  image_pattern = args.pattern or IMAGE_PATTERN
  model = MODELS[args.model](mode='extract', **kwargs)
  input_shape = model.input_shape[0]

  preprocess_image_func = lambda img: model.preprocess_image(img)
  image_generator, steps = generate_images(
    args.images, 
    batch_size=args.batch_size, 
    resize=input_shape,
    func=preprocess_image_func,
    pattern=image_pattern,
    random_crops=args.random_crops
  )

  if args.random_crops:
    print('Random crops were specified, splitting generator into chunks...')
    predict_model = model.build()
    chunks = steps / float(args.random_crops)
    chunks_ceil = int(np.ceil(chunks))
    for n in xrange(chunks_ceil):

      print('Processing chunk {} / {}'.format(n, chunks_ceil-1))

      if not chunks.is_integer() and n == chunks_ceil-1:
        #last chunk is not full
        chunk_steps = steps % args.random_crops
      else:
        chunk_steps = args.random_crops

      generator = islice(image_generator, chunk_steps)
      cnn_f, finetune_f = predict_model.predict_generator(
        generator, chunk_steps, verbose=True)

      cnn_f = cnn_f.reshape(
        (-1, args.random_crops) + cnn_f.shape[1:])
      finetune_f = finetune_f.reshape(
        (-1, args.random_crops) + finetune_f.shape[1:])

      cnn_features = os.path.join(output_dir, 
        '{}_cnn_features.npy'.format(n))
      np.save(cnn_features, np.array(cnn_f))

      finetune_features = os.path.join(output_dir, 
        '{}_finetune_features.npy'.format(n))
      np.save(finetune_features, np.array(finetune_f))

    concatenate_without_loading(output_dir, 
      '^[0-9]*_cnn_features.npy', 
      'random_crop_cnn_features.npy')

    concatenate_without_loading(output_dir, 
      '^[0-9]*_finetune_features.npy', 
      'random_crop_finetune_features.npy')

  else:
    cnn_f, finetune_f = model.build() \
      .predict_generator(image_generator, steps, verbose=True)

    cnn_features = os.path.join(output_dir, 'cnn_features.npy')
    np.save(cnn_features, np.array(cnn_f))

    finetune_features = os.path.join(output_dir, 'finetune_features.npy')
    np.save(finetune_features, np.array(finetune_f))

    
if __name__ == '__main__':
  main()
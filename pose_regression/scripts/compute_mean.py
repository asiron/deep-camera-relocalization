from __future__ import print_function

import argparse, os
import numpy as np

from tqdm import tqdm
from itertools import takewhile

from ..utils import generate_images_from_filenames, make_dir

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--images',
    help='List of paths to images')
  parser.add_argument('-o', '--output', 
    help='Path to an output directory')
  parser.add_argument('-b', '--batch-size', type=int, default=100,
    help='Batch size for image feature extraction')
  parser.add_argument('--resize', type=str, required=True,
    help='Size to resize the image')
  parser.add_argument('-r', '--random-crops', type=int, default=0,
    help='Random crops per image')
  parser.add_argument('-s', '--seed', type=int, default=42,
    help='PRNG seed')

  args = parser.parse_args()

  np.random.seed(args.seed)

  output_dir = args.output
  make_dir(output_dir)

  new_height, new_width = map(int, args.resize.split('x'))
  print('Resizing the images to: {}x{}'.format(new_height, new_width))

  with open(args.images, 'rt') as f:

    image_filenames = f.read().splitlines()[0].split(' ')[:-1]
    image_count = len(image_filenames)
    if args.random_crops:
      total = (image_count / args.batch_size) * args.random_crops
    else:
      total = (image_count / args.batch_size)

    image_generator, _ = generate_images_from_filenames(
      image_filenames, 
      batch_size=args.batch_size, 
      resize=new_height,
      random_crops=args.random_crops
    )

    sums = sum(b.sum(axis=0)
     for b in tqdm(takewhile(lambda b: b is not None, image_generator), total=total))
    mean = sums / image_count

    meanfile = os.path.join(args.output, 'meanfile.npy')
    np.save(meanfile, mean)
    
if __name__ == '__main__':
  main()
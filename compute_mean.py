#!/usr/bin/env
import argparse, os
import numpy as np
from tqdm import tqdm

from utils import generate_images_from_filenames, make_dir

from itertools import takewhile

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

  args = parser.parse_args()
  output_dir = args.output
  make_dir(output_dir)

  new_height, new_width = map(int, args.resize.split('x'))
  print 'Resizing the images to: {}x{}'.format(new_height, new_width)

  with open(args.images, 'rt') as f:

    image_filenames = f.read().splitlines()[0].split(' ')[:-1]

    image_generator, _ = generate_images_from_filenames(
      image_filenames, 
      batch_size=args.batch_size, 
      resize=(new_height, new_width)
    )

    sums = ((b.sum(axis=0), b.shape[0]) 
      for b in tqdm(takewhile(lambda b: b is not None, image_generator)))
    sums = reduce(lambda x,y: (x[0]+y[0], x[1]+y[1]), sums)
    mean = sums[0] / sums[1]

    meanfile = os.path.join(args.output, 'meanfile.npy')
    np.save(meanfile, mean)

    
if __name__ == '__main__':
  main()
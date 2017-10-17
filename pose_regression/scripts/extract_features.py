import numpy as np
import argparse, os

from tqdm import tqdm
from itertools import izip, takewhile

from ..cnn.googlenet import GoogleNet
from ..cnn.inception_resnet_v2 import InceptionResNetV2

from ..utils import make_dir, generate_images, IMAGE_PATTERN

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

  args = parser.parse_args()

  output_dir = args.output
  make_dir(output_dir)

  image_pattern = args.pattern or IMAGE_PATTERN
  model = MODELS[args.model](dataset=args.dataset, 
    mode='extract', meanfile=args.meanfile)
  input_shape = model.input_shape[:2]

  preprocess_image_func = lambda img: model.preprocess_image(img)
  image_generator, steps = generate_images(
    args.images, 
    batch_size=args.batch_size, 
    resize=input_shape,
    func=preprocess_image_func,
    pattern=image_pattern
  )

  outputs = model.model.predict_generator(image_generator, steps, verbose=True)
  cnn_f, finetune_f = outputs[0], outputs[1:]

  cnn_features = os.path.join(output_dir, 'cnn_features.npy')
  np.save(cnn_features, np.array(cnn_f))

  if len(finetune_f) == 1:
    finetune_features = os.path.join(output_dir, 'finetune_features.npy')
    np.save(finetune_features, np.array(finetune_f[0]))
  else:
    finetune_features_head_1 = os.path.join(output_dir, 'finetune_features_head_1.npy')
    finetune_features_head_2 = os.path.join(output_dir, 'finetune_features_head_2.npy')
    finetune_features_head_3 = os.path.join(output_dir, 'finetune_features_head_3.npy')
    np.save(finetune_features_head_1, np.array(finetune_f[0]))   
    np.save(finetune_features_head_2, np.array(finetune_f[1]))   
    np.save(finetune_features_head_3, np.array(finetune_f[2]))   

    
if __name__ == '__main__':
  main()
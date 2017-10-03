#!/usr/bin/env python

import re
import numpy as np
from keras.applications.vgg16 import VGG16

CAFFE_CONVERTED_WEIGHTS_PATH = 'caffe-weights/weights.npy'

pattern

def main():
  vgg16 = VGG16(weights=None, include_top=True)

  weights = np.load(CAFFE_CONVERTED_WEIGHTS_PATH)

re.findall('block(\d+)_conv(\d+)', lnames[4])  



if __name__ == '__main__':
  main()
import scipy.misc
import numpy as np
import uuid, os

def load_and_process(image_filename, resize, random_crop=False, save=True):
  '''
  Loads the image in RGB mode (channels last) and preprocesses it
  '''
  image = scipy.misc.imread(image_filename, mode='RGB')
  image = preprocess_image(image, resize, random_crop=random_crop)
  
  if save:
    dirname = os.path.join(os.path.dirname(image_filename), 'crops')
    if not os.path.exists(dirname):
      os.makedirs(dirname)
    basename = os.path.basename(image_filename).split('.')[0]
    path = os.path.join(dirname, basename)
    if random_crop:
      new_filename = '{}-{}.png'.format(path, uuid.uuid4())
    else:
      new_filename = '{}-center.png'.format(path)
    scipy.misc.imsave(new_filename, image)

  return image

def preprocess_image(img, resize, random_crop=False, 
  random_crop_pixel_variance=20):
  '''
  Resizes, crops and scales the image to 
  a desired (new_height, new_width) format

  If random_crop is Truem then it will first resize to a square with an
  appropriate reduced size by a factor of `reduce_area` and then perform
  a uniform crop
  '''
  if img.shape[0] > img.shape[1]:
    raise ValueError('Image is not in HW format (Height, Width')

  # if random_crop:
  #   img = crop(img, img.shape[0], crop_type='center')
  #   resized_length = new_size + random_crop_pixel_variance
  #   img = scipy.misc.imresize(img, (resized_length, resized_length))
  #   img = crop(img, new_size, crop_type='random')
  # else:
  aspect_ratio = img.shape[1] / float(img.shape[0])
  resized_height = resize[0]
  resized_width = int(resized_height * aspect_ratio)
  img = scipy.misc.imresize(img, (resized_height, resized_width))
  img = crop(img, resize, crop_type='center')

  return img.astype(float)

def crop(img, new_size, crop_type='random'):
  '''Center crops the image to a desired (new_height, new_width) format'''
  y, x = img.shape[:2]
  if crop_type == 'random':
    startx = np.random.randint(0, x - new_size[1])
    starty = np.random.randint(0, y - new_size[0])
  elif crop_type == 'center':
    startx = x // 2 - (new_size[1] // 2)
    starty = y // 2 - (new_size[0] // 2)    
  return img[starty:starty+new_size[0], startx:startx+new_size[1]]

def scale_image(img):
  '''Scales images s.t. each pixel is in [-1, 1] range'''
  img /= 255.0
  img -= 0.5
  img *= 2
  return img

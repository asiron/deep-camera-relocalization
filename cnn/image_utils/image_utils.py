import scipy

def load_and_process(image_filename, resize):
  '''
  Loads the image in RGB mode (channels last) and preprocesses it
  '''
  image = scipy.misc.imread(image_filename, mode='RGB')
  return preprocess_image(image, *resize)

def preprocess_image(img, new_height, new_width):
  '''
  Resizes, crops and scales the image to a desired (new_height, new_width) format
  '''
  if img.shape[0] > img.shape[1]:
    raise ValueError('Image is not in HW format (Height, Width')

  resized_height = new_height
  resized_width = int(new_width * (img.shape[1] / float(img.shape[0])))
  
  img = scipy.misc.imresize(img, (resized_height, resized_width))
  img = center_crop(img, new_height, new_width)
  img = img.astype(float)
  #img = scale_image(img)
  return img

def center_crop(img, new_height, new_width):
  '''
  Center crops the image to a desired (new_height, new_width) format
  '''
  y, x = img.shape[:2]
  startx = x // 2 - (new_width  // 2)
  starty = y // 2 - (new_height // 2)    
  return img[starty:starty+new_height, startx:startx+new_width]

def scale_image(img):
  '''
  Scales images s.t. each pixel is in [-1, 1] range
  '''
  img /= 255.0
  img -= 0.5
  img *= 2
  return img
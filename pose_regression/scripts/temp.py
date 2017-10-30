
def search_files(directory, pattern):
  return [os.path.join(directory, f) 
    for f in os.listdir(directory) if re.search(pattern, f)]

def get_mmaps(directory, pattern):
  return search_files(directory, pattern)
  
def delete_mmaps(directory, pattern):
  mmap_files = get_mmaps(directory, pattern)
  delete = lambda f: os.remove(f)  
  map(delete, mmap_files)

def load_mmaps(directory, pattern):
  mmap_files = get_mmaps(directory, pattern)
  load = lambda f: np.load(f, mmap_mode='r')  
  return map(load, mmap_files)

def concatenate_without_loading(dir, pattern=None, filename=None):

  if pattern is None or filename is None:
  	raise ValueError('Pattern and filename have to be defined!')

  mmaps = load_mmaps(dir, pattern)
  total_size = reduce(lambda x,y: x + y.shape[0], mmaps, 0)

  merged_shape = (total_size,) + mmaps[0].shape[1:]
  merged_shape_path = os.path.join(dir, filename)
  merged_mmap = open_memmap(merged_shape_path,
    dtype=np.float32, mode='w+', shape=merged_shape)

  start_idx, end_idx = 0, 0
  for n, mmap in enumerate(mmaps):

  	end_idx = start_idx + len(mmap)
    merged_mmap[start_idx:end_idx] = f_mmap
    start_idx = end_idx

  del merged_mmap
  #delete_mmaps(dir, pattern)

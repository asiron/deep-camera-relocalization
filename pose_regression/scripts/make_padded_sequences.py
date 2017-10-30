from __future__ import print_function

import numpy as np
from numpy.lib.format import open_memmap

import argparse, os, re

from ..utils import (
  pad_sequences, load_labels, make_dir, 
  make_stateful_sequences,
  make_standard_sequences)

def load_dataset(features_files, labels_files):
  features = [np.squeeze(np.load(f, mmap_mode='r')) for f in features_files]
  labels   = [np.squeeze(load_labels(l)) for l in labels_files]
  return features, labels

def search_files(directory, pattern):
  return [os.path.join(directory, f) 
    for f in os.listdir(directory) if re.search(pattern, f)]

def get_mmaps(directory, feature_type, dataset_type):
  pattern = '^[0-9].*_{feature_type}_{dataset_type}_{{labels_or_features}}'.format(
  feature_type=feature_type, dataset_type=dataset_type)
  
  features_pattern = pattern.format(labels_or_features='features') 
  labels_pattern = pattern.format(labels_or_features='labels') 
  
  features_mmap_files = search_files(directory, features_pattern)
  labels_mmap_files = search_files(directory, labels_pattern)
  
  return features_mmap_files, labels_mmap_files

def delete_mmaps(directory, feature_type, dataset_type):
  features_mmap_files, labels_mmap_files = get_mmaps(
    directory, feature_type, dataset_type)
  
  delete = lambda f: os.remove(f)
  
  map(delete, features_mmap_files)
  map(delete, labels_mmap_files)


def load_mmaps(directory, feature_type, dataset_type):
  features_mmap_files, labels_mmap_files = get_mmaps(
    directory, feature_type, dataset_type)
  
  load = lambda f: np.load(f, mmap_mode='r')
  
  features_mmaps = map(load, features_mmap_files)
  labels_mmaps = map(load, labels_mmap_files)
  return features_mmaps, labels_mmaps

def save_sequences(arr, directory, sequence_type='stateful',
  dataset_type='train', feature_type='cnn',
  features_or_labels='features'):

  seq_output_path = os.path.join(directory, '{}_{}_{}_{}_seqs.npy'
    .format(sequence_type, feature_type, dataset_type, features_or_labels))
  print('Saving {} sequences...'.format(seq_output_path))
  np.save(seq_output_path, arr)

def concatenate_without_loading(dir, feature_type, dataset_type):
    # load saved arrays as mmapped arrays
  features_mmaps, labels_mmaps = load_mmaps(dir, feature_type, dataset_type)
  print([x.shape for x in features_mmaps])
  print([x.shape for x in labels_mmaps])
  total_num_seqs = reduce(lambda x,y: x + y.shape[0], features_mmaps, 0)

  final_features_shape = (total_num_seqs,) + features_mmaps[0].shape[1:]
  final_standard_features_path = os.path.join(dir,
    'standard_{}_{}_features_seqs.npy'.format(feature_type, dataset_type))
  final_standard_features = open_memmap(final_standard_features_path,
    dtype=np.float32, mode='w+', shape=final_features_shape)

  final_labels_shape = (total_num_seqs,) + labels_mmaps[0].shape[1:]
  final_standard_labels_path = os.path.join(dir,
    'standard_{}_{}_labels_seqs.npy'.format(feature_type, dataset_type))
  final_standard_labels = open_memmap(final_standard_labels_path,
    dtype=np.float32, mode='w+', shape=final_labels_shape)

  for n, (f_mmap, l_mmap) in enumerate(zip(features_mmaps, labels_mmaps)):

    from_idx = n * len(features_mmaps[0])
    to_idx = (n+1) * len(features_mmaps[0])
    final_standard_features[from_idx:to_idx] = f_mmap
    final_standard_labels[from_idx:to_idx]   = l_mmap

  del final_standard_features, final_standard_labels
  #delete_mmaps(dir, feature_type, dataset_type)

def process_dataset(features_files, labels_files, dataset_type='train',
                   sequence_type='stateful', feature_type='cnn', output_dir=None,
                   **kwargs):
  
  print('Loading {} sequences...'.format(dataset_type))
  features = [np.squeeze(np.load(f, mmap_mode='r')) for f in features_files]
  labels   = [np.squeeze(load_labels(l)) for l in labels_files]

  print('Preparing {} sequences...'.format(dataset_type))
  if sequence_type == 'stateful':
    
    seq_len = kwargs['seq_len']
    batch_size = kwargs['batch_size']

    features_seqs, labels_seqs = make_stateful_sequences(
      features, labels, 
      seq_len=seq_len, 
      batch_size=batch_size)

    save_sequences(features_seqs, output_dir,
      sequence_type=sequence_type,
      feature_type=feature_type,
      dataset_type=dataset_type,
      features_or_labels='features')

    save_sequences(labels_seqs, output_dir,
      sequence_type=sequence_type,
      feature_type=feature_type,
      dataset_type=dataset_type,
      features_or_labels='labels')

  elif sequence_type == 'standard':

    subseq_len = kwargs['subseq_len']
    
    for n, (fs, ls) in enumerate(make_standard_sequences(features, labels,
      subseq_len=subseq_len)):

      sequence_type = '{:02d}_standard'.format(n)

      save_sequences(fs, output_dir,
        sequence_type=sequence_type,
        feature_type=feature_type,
        dataset_type=dataset_type,
        features_or_labels='features')

      save_sequences(ls, output_dir,
        sequence_type=sequence_type,
        feature_type=feature_type,
        dataset_type=dataset_type,
        features_or_labels='labels')

    concatenate_without_loading(output_dir, feature_type, dataset_type)

def main():
  parser = argparse.ArgumentParser()
  
  parser.add_argument('-tf', '--train-features', nargs='+', required=True,
    help='Path to a numpy array with training features')
  parser.add_argument('-tl', '--train-labels', nargs='+', required=True, 
    help='Path to a directory with training labels')

  parser.add_argument('-vf', '--val-features', nargs='+', required=True,
    help='Path to a numpy array with validation features')
  parser.add_argument('-vl', '--val-labels', nargs='+', required=True, 
    help='Path to a directory with validation labels')

  parser.add_argument('--feature-type', required=True, choices=['cnn', 'finetune'],
    help='Type of features being preprocessed')

  parser.add_argument('-o', '--output', required=True, 
    help='Path to an output dir where numpy arrays should be saved')
 
  parser.add_argument('--seq-len',    type=int, help='Sequence length fir a stateful LSTM')
  parser.add_argument('--batch-size', type=int, help='Batch size for a stateful LSTM')
  parser.add_argument('--subseq-len', type=int, help='Sub-sequence length for standard LSTM')

  parser.add_argument('--type', choices=['stateful', 'standard', 'regressor'],
    help='Output the seqences into Stateful or Stateless LSTM input shape')

  args = parser.parse_args()

  make_dir(args.output)

  if args.type == 'stateful' and not (args.batch_size and args.seq_len):
    raise ValueError('Batch size and sequence length have to be specified' + 
                     'in stateful LSTM mode')

  elif args.type == 'standard' and not args.subseq_len:
    raise ValueError('Sub-sequence length has to be specified' + 
                     'in standard LSTM mode')

  kwargs = {k: vars(args)[k] for k in ('batch_size', 'subseq_len', 'seq_len')}

  process_dataset(args.train_features, args.train_labels, dataset_type='train',
                  sequence_type=args.type, feature_type=args.feature_type, 
                  output_dir=args.output, **kwargs)

  process_dataset(args.val_features, args.val_labels, dataset_type='val',
                  sequence_type=args.type, feature_type=args.feature_type, 
                  output_dir=args.output, **kwargs)

  print('Done!')

if __name__ == '__main__':
  main()
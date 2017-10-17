import numpy as np
import argparse, os

from ..utils import prepare_sequences, load_labels, make_dir

def main():
  parser = argparse.ArgumentParser()
  
  parser.add_argument('-tl', '--train-labels', nargs='+', required=True, 
    help='Path to a directory with training labels')
  parser.add_argument('-tf', '--train-features', nargs='+', required=True,
    help='Path to a numpy array with training features')

  parser.add_argument('-vl', '--val-labels', nargs='+', required=True, 
    help='Path to a directory with validation labels')
  parser.add_argument('-vf', '--val-features', nargs='+', required=True,
    help='Path to a numpy array with validation features')

  parser.add_argument('-o', '--output', required=True, 
    help='Path to an output dir where numpy arrays should be saved')
 
  parser.add_argument('--seq-len', type=int, required=True,
    help='Sequence length')

  args = parser.parse_args()

  train_features_arr = [np.squeeze(np.load(f)) for f in args.train_features]
  train_labels_arr   = [load_labels(l) for l in args.train_labels]

  val_features_arr = [np.squeeze(np.load(f)) for f in args.val_features]
  val_labels_arr   = [load_labels(l) for l in args.val_labels]

  print 'Preparing training sequences...'
  train_features_seqs, train_labels_seqs = prepare_sequences(
    train_features_arr, 
    train_labels_arr, 
    args.seq_len)

  print 'Preparing validation sequences...'
  val_features_seqs, val_labels_seqs = prepare_sequences(
    val_features_arr, 
    val_labels_arr, 
    args.seq_len)

  print 'Saving...'

  make_dir(args.output)

  tf_seq_output_path = os.path.join(args.output, 'train_features_seqs.npy')
  np.save(tf_seq_output_path, train_features_seqs)

  tl_seq_output_path = os.path.join(args.output, 'train_labels_seqs.npy')
  np.save(tl_seq_output_path, train_labels_seqs)

  vf_seq_output_path = os.path.join(args.output, 'val_features_seqs.npy')
  np.save(vf_seq_output_path, val_features_seqs)

  vl_seq_output_path = os.path.join(args.output, 'val_labels_seqs.npy')
  np.save(vl_seq_output_path, val_labels_seqs)

  print 'Done!'


if __name__ == '__main__':
  main()
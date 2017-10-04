import quaternion, argparse, re, os
import numpy as np

'''
Takes a file with 4x4 Transformation matrix and converts it to (frame_idx, pos, quat)
Resulting file is saved at the same place with name -> pos_{frame_idx}.txt in order
to be consistent with my original dataset

Run the command with xargs for 7places dataset in root dir of the dataset:

$ find . -regex ".*frame-[0-9]*\.pose\.txt$" \
  | xargs -l -n1 -P 12 python convert_pose_file.py

'''

patt = re.compile("[^\t\ \r\n]+")

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('pose_file', help='Path to a file with pose')
  args = parser.parse_args()

  with open(args.pose_file) as f:
    lines = f.readlines()
    matrix = map(lambda l: map(float, patt.findall(l)), lines)
    matrix = np.array(matrix)
    rot, pos = matrix[0:3,0:3], matrix[0:3,3]
    quat = quaternion.from_rotation_matrix(rot)
    quat = quaternion.as_float_array(quat)
    
    output_file_dir = os.path.dirname(args.pose_file)
    pose_file_basename = os.path.basename(args.pose_file).split('.')[0]
    frame_idx = pose_file_basename.split('-')[1]
   
    output_filepath = os.path.join(
      output_file_dir,
      'pos_{}.txt'.format(frame_idx))

    out = np.concatenate(([float(frame_idx)], pos, quat))
    np.savetxt(output_filepath, out.reshape((1,-1)), delimiter=',')


if __name__ == '__main__':
  main()
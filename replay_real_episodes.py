#!/usr/bin/env python3
import os
import h5py
import argparse
from robot.real_env import make_real_env

def main(args):
    # Read args
    dataset_dir = args['dataset_dir']
    episode_idx = args['episode_idx']
    dataset_name = f'episode_{episode_idx}'

    # Open episode file
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Episode file does not exist at \n{dataset_path}\n')
        exit()

    # Connect to real robot
    env = make_real_env()
    env.reset()

    # Load actions
    with h5py.File(dataset_path, 'r') as root: actions = root['/action'][()]

    # Replay actions
    for action in actions: env.step(action, True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default="data/robot_move", action='store', type=str, help='Dataset dir.', required=False)
    parser.add_argument('--episode_idx', default=0, action='store', type=int, help='Episode index.', required=False)
    main(vars(parser.parse_args()))



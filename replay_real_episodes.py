#!/usr/bin/env python3
import os
import h5py
import argparse
from robot.real_env import make_real_env
from robot.constants_robot import JOINT_NAMES, PUPPET_GRIPPER_JOINT_OPEN

def main(args):
    # Read args
    dataset_dir = args['dataset_dir']
    if not dataset_dir: dataset_dir = "data/robot_move"
    episode_idx = args['episode_idx']
    if not episode_idx: episode_idx = 0
    dataset_name = f'episode_{episode_idx}'

    # Open episode
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Episode file does not exist at \n{dataset_path}\n')
        exit()

    # Load actions
    with h5py.File(dataset_path, 'r') as root:
        actions = root['/action'][()]

    # Connect to real robot
    env = make_real_env(init_node=True)
    env.reset()

    # Replay actions
    for action in actions: env.step(action, True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=False)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', required=False)
    main(vars(parser.parse_args()))



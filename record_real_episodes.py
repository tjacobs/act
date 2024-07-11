#!/usr/bin/env python3
import os
import time
import h5py
import argparse
import numpy as np
from tqdm import tqdm

from robot.constants_robot import DT, NUM_JOINTS, TASK_CONFIGS
from robot.recorder import Recorder
from robot.robot_utils import ImageRecorder
from robot.real_env import make_real_env

def main(args):
    # Load config from constants_robot task config
    task_config = TASK_CONFIGS[args['task_name']]
    dataset_dir = task_config['dataset_dir']
    max_timesteps = task_config['episode_len']
    camera_names = task_config['camera_names']

    # Start with correct episode number
    if args['episode_idx'] is not None: episode_idx = args['episode_idx']
    else:                               episode_idx = get_auto_index(dataset_dir)

    # Capture one episode, retry if it fails
    dataset_name = f'episode_{episode_idx}'
    overwrite = False
    while True:
        is_healthy = capture_one_episode(DT, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite)
        if is_healthy: break

# Record one episode of data from the real robot
def capture_one_episode(dt, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite):
    # Print
    print(f'Recording: {dataset_name}')

    # Set up connection to real robot
    env = make_real_env(camera_names)

    # Check files
    if not os.path.isdir(dataset_dir): os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isfile(dataset_path) and not overwrite:
        print(f'Dataset already exist at \n{dataset_path}\nHint: set overwrite to True.')
        exit()

    # Record data
    ts = env.reset()
    timesteps = [ts]
    actions = []
    actual_dt_history = []
    for t in tqdm(range(max_timesteps)):
        t0 = time.time()
        action = env.get_action() # Read robot joint positions
        print(action)
        t1 = time.time()
        ts = env.step(action, False) # Get observations from camera
        t2 = time.time()
        timesteps.append(ts)
        actions.append(action)
        actual_dt_history.append([t0, t1, t2])

    # Check if recording was smooth
    #freq_mean = print_dt_diagnosis(actual_dt_history)
    #print(freq_mean)
    #if freq_mean < 42: return False

    """
    For each timestep:

    observations
    - qpos                  (NUM_JOINTS,)         'float64'
    - qvel                  (NUM_JOINTS,)         'float64'
    - effort                (NUM_JOINTS,)         'float64'
    - images
        - cam_high          (480, 640, 3) 'uint8'
        - cam_low           (480, 640, 3) 'uint8'
    
    action                  (NUM_JOINTS,)         'float64'
    """

    # Init
    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/action': [],
    }
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []

    # Put in data dictionary
    while actions:
        action = actions.pop(0)
        ts = timesteps.pop(0)
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/action'].append(action)
        for cam_name in camera_names: data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

    # Save to HDF5 file
    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        root.attrs['sim'] = False
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in camera_names:
            _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8', chunks=(1, 480, 640, 3), )
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
        _ = obs.create_dataset('qpos', (max_timesteps, NUM_JOINTS))
        _ = obs.create_dataset('qvel', (max_timesteps, NUM_JOINTS))
        _ = root.create_dataset('action', (max_timesteps, NUM_JOINTS))
        for name, array in data_dict.items(): root[name][...] = array
        print(f"Saved: {dataset_path}.hdf5")
    #print(f'Saving: {time.time() - t0:.1f} secs')

    # Good
    return True


def get_auto_index(dataset_dir, dataset_name_prefix = '', data_suffix = 'hdf5'):
    max_idx = 1000
    if not os.path.isdir(dataset_dir): os.makedirs(dataset_dir)
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def print_dt_diagnosis(actual_dt_history):
    actual_dt_history = np.array(actual_dt_history)
    get_action_time = actual_dt_history[:, 1] - actual_dt_history[:, 0]
    step_env_time = actual_dt_history[:, 2] - actual_dt_history[:, 1]
    total_time = actual_dt_history[:, 2] - actual_dt_history[:, 0]
    dt_mean = np.mean(total_time)
    dt_std = np.std(total_time)
    freq_mean = 1 / dt_mean
    print(f'Avg freq: {freq_mean:.2f} Get action: {np.mean(get_action_time):.3f} Step env: {np.mean(step_env_time):.3f}')
    return freq_mean


def debug():
    print(f'====== Debug mode ======')
    recorder = Recorder()
    #image_recorder = ImageRecorder(init_node=False, is_debug=True)
    while True:
        time.sleep(1)
        recorder.print_diagnostics()
        #image_recorder.print_diagnostics()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=None, required=False)
    main(vars(parser.parse_args()))
    #debug()



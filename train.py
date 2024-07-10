#!/usr/bin/env python3
import os
import torch
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from policy import ACTPolicy
from constants import DT
from utils import load_data
from utils import compute_dict_mean, set_seed, detach_dict
from visualize_episodes import save_videos
from sim_env import BOX_POSE

def main(args):
    # Set seed
    set_seed(1)

    # Get command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # Look up task config based on sim or real
    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from robot.constants_robot import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]

    # Get task parameters
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # Set policy_config parameters
    state_dim = 14
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         }
    else:
        raise NotImplementedError

    # Set config parameters, including adding policy_config
    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim
    }

    # Policy evaluation time?
    if is_eval:
        # Evaluate all policies in policy list
        policies = [f'policy_best.ckpt']
        results = []
        for policy in policies:
            success_rate, avg_return = evaluate_policy(config, policy, save_episode=True)
            results.append([policy, success_rate, avg_return])

        # Print and done
        for policy, success_rate, avg_return in results:
            print(f'{policy}: {success_rate=} {avg_return=}')
        print()
        exit()

    else:
        # Policy training time
        train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val)

        # Save dataset stats
        if not os.path.isdir(ckpt_dir): os.makedirs(ckpt_dir)
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'wb') as f: pickle.dump(stats, f)

        # Train policy
        best_ckpt_info = train_policy(train_dataloader, val_dataloader, config)
        best_epoch, min_val_loss, best_state_dict = best_ckpt_info

        # Save best checkpoint
        ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
        torch.save(best_state_dict, ckpt_path)
        print(f'Best checkpoint, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


# Evaluate a policy by running it
def evaluate_policy(config, ckpt_name, save_episode=True):
    # Set seed to something different
    set_seed(1000)

    # Load config
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']

    # Select camera
    onscreen_cam = 'angle'

    # Load policy
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    print("Making policy:")
    policy = make_policy(policy_class, policy_config)
    print("Loading:")
    if torch.cuda.is_available():
        loading_status = policy.load_state_dict(torch.load(ckpt_path, weights_only=True))
    else:
        loading_status = policy.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=torch.device('cpu')))
    print(loading_status)
    if torch.cuda.is_available(): policy.cuda()

    # Eval policy
    policy.eval()
    print(f'Loaded: {ckpt_path}')

    # Load stats
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # Define pre-process and post-process functions
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # Load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers
        from aloha_scripts.real_env import make_real_env
        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    # Queries
    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    # Increase max_timesteps for real-world tasks
    max_timesteps = int(max_timesteps * 1)

    # Roll out
    num_rollouts = 2
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### Set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # Used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # Used in sim reset

        # Reset and get first timestep
        ts = env.reset()

        ### Onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### Evaluation loop
        if temporal_agg:
            if torch.cuda.is_available():
                all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()
            else:
                all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim])

        ### Zero
        if torch.cuda.is_available():
            qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        else:
            qpos_history = torch.zeros((1, max_timesteps, state_dim))
        image_list = [] # For visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### Update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### Process previous timestep to get qpos and image_list
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                if torch.cuda.is_available():
                    qpos = torch.from_numpy(qpos).float().unsqueeze(0).cuda()
                else:
                    qpos = torch.from_numpy(qpos).float().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)

                ### Query policy for what action to do, inference
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        if torch.cuda.is_available():
                            exp_weights = torch.from_numpy(exp_weights).unsqueeze(dim=1).cuda()
                        else:
                            exp_weights = torch.from_numpy(exp_weights).unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                else:
                    raise NotImplementedError

                ### Post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### Step the environment, sim or real
                ts = env.step(target_qpos)

                ### Add data for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            # Close visualization
            plt.close()

        # Open real robot grippers when done
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # Open
            pass

        # Stats
        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        # Save video
        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    # End stats
    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'
    print(summary_str)

    # Save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    # Done
    return success_rate, avg_return


# Train the policy
def train_policy(train_dataloader, val_dataloader, config):
    # Get config
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    # Set seed
    set_seed(seed)

    # Create the policy and optimizer
    policy = make_policy(policy_class, policy_config)
    if torch.cuda.is_available():
        policy.cuda()
    optimizer = make_optimizer(policy_class, policy)
    print("")

    # Init
    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None

    # For each epoch, show progress bar with ETA
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}:', end=" ")

        # Validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            # Loss
            epoch_val_loss = epoch_summary['loss']

            # Best loss?
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))

        # Print validation loss
        print(f'Validation loss: {epoch_val_loss:.2f}', end="\t")
        summary_string = ''
        for k, v in epoch_summary.items(): summary_string += f'{k}: {v.item():.3f} '
        #print(summary_string)

        # Training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            # Forward
            forward_dict = forward_pass(data, policy)

            # Backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))

        # Print training loss
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Training loss: {epoch_train_loss:.2f}')
        summary_string = ''
        for k, v in epoch_summary.items(): summary_string += f'{k}: {v.item():.3f} '
        #print(summary_string)

        # Save checkpoint
        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    # Save checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    # Save best checkpoint
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)

    # Print
    print(f'Training finished:\nSeed {seed}, validation loss {min_val_loss:.6f} at epoch {best_epoch}')

    # Save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    # Done
    return best_ckpt_info


# Get camera image from real or sim
def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    if torch.cuda.is_available():
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    else:
        curr_image = torch.from_numpy(curr_image / 255.0).float().unsqueeze(0)
    return curr_image


# Run the policy
def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    if torch.cuda.is_available():
        image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad)


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # Save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    #print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval',            default=False,         action='store_true')
    parser.add_argument('--onscreen_render', default=False,         action='store_true')
    parser.add_argument('--temporal_agg',    default=False,         action='store_true')
    parser.add_argument('--ckpt_dir',        default='trained',     action='store', type=str,   help='ckpt_dir')
    parser.add_argument('--policy_class',    default='ACT',         action='store', type=str,   help='policy_class')
    parser.add_argument('--task_name',       default='robot_move',  action='store', type=str,   help='task_name')
    parser.add_argument('--batch_size',      default=8,             action='store', type=int,   help='batch_size')
    parser.add_argument('--seed',            default=0,             action='store', type=int,   help='seed')
    parser.add_argument('--num_epochs',      default=2000,          action='store', type=int,   help='num_epochs')
    parser.add_argument('--lr',              default=1e-5,          action='store', type=float, help='lr')
    parser.add_argument('--kl_weight',       default=10,            action='store', type=int,   help='KL Weight')
    parser.add_argument('--chunk_size',      default=100,           action='store', type=int,   help='chunk_size')
    parser.add_argument('--hidden_dim',      default=512,           action='store', type=int,   help='hidden_dim')
    parser.add_argument('--dim_feedforward', default=3200,          action='store', type=int,   help='dim_feedforward')
    main(vars(parser.parse_args()))

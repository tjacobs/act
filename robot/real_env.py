import time
import numpy as np
import collections
import matplotlib.pyplot as plt
import dm_env

from robot.constants_robot import DT, NUM_JOINTS, START_ARM_POSE, MASTER_GRIPPER_JOINT_NORMALIZE_FN, PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN
from robot.constants_robot import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN
from robot.constants_robot import PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
from robot.recorder import Recorder
from robot.robot_utils import ImageRecorder
from robot.robot_utils import setup_master_bot, setup_puppet_bot, move_arms, move_grippers

class RealEnv:
    """
    Environment for real robot

    Action space:      [arm_qpos (NUM_JOINTS)]                       # absolute joint positions

    Observation space:  "qpos": Concat[ left_arm_qpos (NUM_JOINTS) ] # absolute joint positions
                        "qvel": Concat[ left_arm_qvel (NUM_JOINTS) ] # absolute joint velocities (rad)
                        "images": {"cam_high": (480x640x3),          # h, w, c, dtype='uint8'
                                   "cam_low": (480x640x3)}           # h, w, c, dtype='uint8'
    """

    def __init__(self, init_node, setup_robots=True):
        # Create robot joint data reader
        self.recorder = Recorder()
        #self.image_recorder = ImageRecorder(init_node=False)

    def get_observation(self):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos()
        obs['qvel'] = self.get_qvel()
        obs['effort'] = self.get_effort()
        obs['images'] = self.get_images()
        return obs

    def get_qpos(self):
        qpos = self.recorder.get_joint_positions()
        #print(qpos)
        return qpos

    def get_qvel(self):
        return [0, 0, 0]

    def get_effort(self):
        return [0, 0, 0]

    def get_images(self):
        return [] #self.image_recorder.get_images()

    def get_reward(self):
        return 0

    def reset(self, fake=False):
        if not fake:
            # Reboot puppet robot gripper motors
            self.puppet_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
            self.puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)
            self._reset_joints()
            self._reset_gripper()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())

    def step(self, action):
        state_len = int(len(action) / 2)
        left_action = action[:state_len]
        right_action = action[state_len:]
        #self.puppet_bot_left.arm.set_joint_positions(left_action[:6], blocking=False)
        #self.puppet_bot_right.arm.set_joint_positions(right_action[:6], blocking=False)
        #self.set_gripper_pose(left_action[-1], right_action[-1])
        #time.sleep(DT)
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())


def get_action(master_bot_left, master_bot_right):
    # Action is NUM_JOINTS joints
    action = np.zeros(NUM_JOINTS) 
    action[0] = 0
    action[1] = 0
    action[2] = 0

    # Arm actions
    #action[:6] = master_bot_left.dxl.joint_states.position[:6]
    #action[7:7+6] = master_bot_right.dxl.joint_states.position[:6]

    # Gripper actions
    #action[6] = MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_left.dxl.joint_states.position[6])
    #action[7+6] = MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_right.dxl.joint_states.position[6])

    return action


def make_real_env(init_node, setup_robots=True):
    env = RealEnv(init_node, setup_robots)
    return env


def test_real_teleop():
    """
    Test bimanual teleoperation and show image observations onscreen.
    It first reads joint poses from both master arms.
    Then use it as actions to step the environment.
    The environment returns full observations including images.

    An alternative approach is to have separate scripts for teleoperation and observation recording.
    This script will result in higher fidelity (obs, action) pairs
    """

    onscreen_render = True
    render_cam = 'cam_left_wrist'

    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name=f'master_right', init_node=False)
    setup_master_bot(master_bot_left)
    setup_master_bot(master_bot_right)

    # setup the environment
    env = make_real_env(init_node=False)
    ts = env.reset(fake=True)
    episode = [ts]
    # setup visualization
    if onscreen_render:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam])
        plt.ion()

    for t in range(1000):
        action = get_action(master_bot_left, master_bot_right)
        ts = env.step(action)
        episode.append(ts)

        if onscreen_render:
            plt_img.set_data(ts.observation['images'][render_cam])
            plt.pause(DT)
        else:
            time.sleep(DT)


if __name__ == '__main__':
    test_real_teleop()


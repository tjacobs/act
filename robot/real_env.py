import time
import collections
import dm_env

from robot.constants_robot import DT, NUM_JOINTS
from robot.recorder import Recorder, ImageRecorder

class RealEnv:
    """
    Environment for real robot

    Action space:      [arm_qpos (NUM_JOINTS)]                         # absolute joint positions

    Observation space:  "qpos": Concat[ robot_joint_pos (NUM_JOINTS) ] # absolute joint positions
                        "qvel": Concat[ robot_joint_vel (NUM_JOINTS) ] # absolute joint velocities (rad)
                        "images": {"cam_1": (480x640x3)}               # h, w, c, dtype='uint8'
    """

    def __init__(self, camera_names):
        # Create robot joint data reader recorder, and camera image recorder
        self.recorder = Recorder()
        self.image_recorder = ImageRecorder(camera_names)

    def get_action(self):
        # Action is NUM_JOINTS number of values
        action = self.recorder.get_joint_positions()
        return action

    def get_observation(self):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos()
        obs['qvel'] = self.get_qvel()
        obs['images'] = self.get_images()
        return obs

    def get_qpos(self):
        qpos = self.recorder.get_joint_positions()
        #print(qpos)
        return qpos

    def get_qvel(self):
        return [0, 0, 0]

    def get_images(self):
        return self.image_recorder.get_images()

    def get_reward(self):
        return 0

    def reset(self):
        # Return first timestep with reward and observation
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())

    def step(self, action, move_robot):
        # Set robot joints to positions of this timestep's action if asked to
        if move_robot:
            self.recorder.set_joint_positions(action)
            time.sleep(DT)

        # Return this timestep with reward and observation
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())


def make_real_env(camera_names = []):
    env = RealEnv(camera_names)
    return env


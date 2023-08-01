import gym
import numpy as np
import pybullet as p
from igibson.robots.fetch import Fetch
#from igibson.robots.robot_locomotor import LocomotorRobot
from igibson.external.pybullet_tools.utils import joints_from_names, set_joint_positions


class Fetch_DD(Fetch):
    def __init__(self,angular=2.0,linear=1.0,**config):
        self.wheel_dim = 2
        self.wheel_axle_half = 0.186
        super(Fetch_DD,self).__init__(reset_joint_pos='tuck',**config)
        
        self.wheel_dim = 2
        self.vel_coefficient = 1.0
        self.locobot_wheel_joint_max = [15.0,15.0]
        self.linear_velocity = linear
        self.angular_velocity = angular
        self.set_up_continuous_action_space()

    def set_up_continuous_action_space(self):
        self.action_high = np.zeros(self.wheel_dim)
        self.action_high[0] = self.linear_velocity
        self.action_high[1] = self.angular_velocity
        self.action_low = -self.action_high
        self.action_space_custom_xyz = gym.spaces.Box(shape=(self.wheel_dim,), low=-1.0, high=1.0, dtype=np.float32)

    

    #post-release added
    def policy_action_to_robot_action(self, action):
        """
        Scale the policy action (always in [-1, 1]) to robot action based on action range

        :param action: policy action
        :return: robot action
        """
        
        action = np.clip(action, self.action_space_custom_xyz.low, self.action_space_custom_xyz.high)

        # de-normalize action to the appropriate, robot-specific scale
        real_action = (self.action_high - self.action_low) / 2.0 * action + (
            self.action_high + self.action_low
        ) / 2.0
        return real_action

    #igibson 2.0.3 version control
    def apply_action_old(self,action):
        real_action = self.policy_action_to_robot_action(action)
        lin_vel, ang_vel = real_action
            
        left_wheel_ang_vel = (lin_vel - ang_vel * self.wheel_axle_half) / self.wheel_radius
        right_wheel_ang_vel = (lin_vel + ang_vel * self.wheel_axle_half) / self.wheel_radius

        v_v = [right_wheel_ang_vel,left_wheel_ang_vel]
        
        joints = self._joints.values()
        
        for i,j in enumerate(joints):
            v = self.vel_coefficient *v_v[i]
            v = np.clip(v, -j.max_velocity, j.max_velocity)
            p.setJointMotorControl2(j.body_id, j.joint_id, p.VELOCITY_CONTROL, targetVelocity=v)
            #stop after the first two wheeled joints
            if i == 1:
                break
            
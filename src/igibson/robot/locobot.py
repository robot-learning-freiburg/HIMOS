from igibson.robots.locobot import Locobot
from igibson.external.pybullet_tools.utils import joints_from_names, set_joint_positions
import numpy as np
import gym
import pybullet as p
class Locobot_DD(Locobot):
    def __init__(self,angular=1.0,linear=0.5, **config):
        super(Locobot_DD,self).__init__(**config)
        self.wheel_dim = 2
        self.linear_velocity = linear
        self.angular_velocity = angular
        self.set_up_continuous_action_space()
        self.wheel_axle_half = 0.115  # half of the distance between the wheels 230
        self.wheel_radius_mod = 0.038  # radius of the wheels
        self.vel_coefficient = 1.0
    def set_up_continuous_action_space(self):
        self.action_high = np.zeros(self.wheel_dim)
        self.action_high[0] = self.linear_velocity
        self.action_high[1] = self.angular_velocity
        self.action_low = -self.action_high
        self.action_space_custom_xyz = gym.spaces.Box(shape=(self.wheel_dim,), low=-1.0, high=1.0, dtype=np.float32)

    #added by Fabi
    def policy_action_to_robot_action(self, action):
        """
        Scale the policy action (always in [-1, 1]) to robot action based on action range

        :param action: policy action
        :return: robot action
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # de-normalize action to the appropriate, robot-specific scale
        real_action = (self.action_high - self.action_low) / 2.0 * action + (
            self.action_high + self.action_low
        ) / 2.0
        return real_action

    #igibson 2.0.3 version control
    def apply_action_fabi(self,action):
        real_action = self.policy_action_to_robot_action(action)
        lin_vel, ang_vel = real_action
            
        left_wheel_ang_vel = (lin_vel - ang_vel * self.wheel_axle_half) / self.wheel_radius_mod
        right_wheel_ang_vel = (lin_vel + ang_vel * self.wheel_axle_half) / self.wheel_radius_mod

        v_v = [left_wheel_ang_vel,right_wheel_ang_vel]
        
        joints = self._joints.values()
        #assumed there are only two joints corresponding to the two wheels
        for i,j in enumerate(joints):
            
            v = self.vel_coefficient * v_v[i] * j.max_velocity
            v = np.clip(v, -j.max_velocity, j.max_velocity) #this is assumed to be done in igibson_2.0.3 in Robot_base in the Joint class
            p.setJointMotorControl2(j.body_id, j.joint_id, p.VELOCITY_CONTROL, targetVelocity=v)
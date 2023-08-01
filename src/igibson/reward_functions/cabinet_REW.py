from igibson.reward_functions.reward_function_base import BaseRewardFunction
from igibson.utils.utils import l2_distance
import numpy as np


class CabinetReward(BaseRewardFunction):
    """
    Point goal reward
    Success reward for reaching the goal with the robot's base
    """

    def __init__(self, config):
        super(CabinetReward, self).__init__(config)
        self.cabinet_indices = [0,1,2]
        self.object_values = np.array([111, 26, 177])
    def get_reward(self, task, env, info):

        reward = 0.0

        for i in self.cabinet_indices:
            if task.wanted_plates[i] == 1:
                masked_category = (env.global_map[:, :, 0] == self.object_values[i])
                num_pixel = masked_category.sum()
                if num_pixel > 5:
                    reward += 5.0
                    task.wanted_plates[i] = 0

        return reward

    def reset(self, task, env):
        pass




from igibson.reward_functions.reward_function_base import BaseRewardFunction
from igibson.utils.utils import l2_distance
import numpy as np

class GenericPenaltyReward(BaseRewardFunction):
    """
    DoorOpeningReward
    penelizes the individual distance travelled, invalid action taken and collision during navigation.
    """

    def __init__(self, config):
        super(GenericPenaltyReward, self).__init__(config)

    def get_reward(self, task, env,info):

        reward = 0

        if info["planner_collision"]:
            reward -= 0.1

        if info['penalty']:
            reward -= 2.5
        
        reward -= (info['discount_length'])*0.05
        
        return reward
        
        
    def reset(self, task, env):
        pass

        


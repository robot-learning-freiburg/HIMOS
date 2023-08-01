from igibson.reward_functions.reward_function_base import BaseRewardFunction
from igibson.utils.utils import l2_distance
import numpy as np

class PointGoalReward(BaseRewardFunction):
    """
    Point goal reward
    Success reward for reaching the goal with the robot's base
    """

    def __init__(self, config):
        super(PointGoalReward, self).__init__(config)
        self.object_values = np.array([64,32,12,102,126,140])
        self.cracker_index_conversion = {3:0,4:1,5:2}
        self.min_pix = [29,29,29,29,29,29,29]
        self.dist_tol = self.config.get("dist_tol", 0.5)

    def get_reward(self, task, env,info):
        """
        Check if the distance between the robot's base and the goal
        is below the distance threshold

        :param task: task instance
        :param env: environment instance
        :return: reward
        """

        reward = 0 #-0.0025 #used to be the slack reward

        for i in task.uniq_indices:
            
            if task.wanted_objects[i] == 1:
                masked_category = (env.global_map[:,:,0] == self.object_values[i])
                num_pixel = masked_category.sum()

                if num_pixel > 20:
                    success = l2_distance(env.robots[0].get_position()[:2], task.target_pos_list[i][:2]) < self.dist_tol
                    if success:
                        
                        reward += 10.0
                        task.wanted_objects[i] = 0
                        tr_ps = task.target_pos_list[int(i)]
                        tr_ps = env.mapping.world2map(tr_ps)
                        env.global_map[int(tr_ps[1])-5:int(tr_ps[1])+5,int(tr_ps[0])-3:int(tr_ps[0])+2] = env.mapping.category_found
                        env.cracker_indices.append(self.cracker_index_conversion[i])

        
        return reward
        
        
    def reset(self, task, env):
        pass
        #self.indices = np.argwhere(env.wanted_objects != 0)
        


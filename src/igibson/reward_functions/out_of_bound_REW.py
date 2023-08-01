
from igibson.termination_conditions.termination_condition_base import BaseTerminationCondition

from igibson.reward_functions.reward_function_base import BaseRewardFunction

class OutOfBoundRew(BaseRewardFunction):
    """
    OutOfBound used for navigation tasks in InteractiveIndoorScene
    Episode terminates if the robot goes outside the valid region
    """

    def __init__(self, config):
        super(OutOfBoundRew, self).__init__(config)
        self.fall_off_thresh = self.config.get("fall_off_thresh", 0.03)

    def get_reward(self, task, env,info):
        """
        Return whether the episode should terminate.
        Terminate if the robot goes outside the valid region

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """

        reward = 0.0
        # fall off the cliff of valid region
        #if isinstance(env.scene, InteractiveIndoorScene):
        robot_z = env.robots[0].get_position()[2]
        if robot_z < (env.scene.get_floor_height() - self.fall_off_thresh):
            reward -= 2.0
        
        return reward

    def reset(self, task, env):
        pass
        #self.indices = np.argwhere(env.wanted_objects != 0)
        


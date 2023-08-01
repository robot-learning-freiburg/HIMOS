from igibson.termination_conditions.termination_condition_base import BaseTerminationCondition
from igibson.utils.utils import l2_distance


class PointGoal(BaseTerminationCondition):
    """
    PointGoal used for PointNavFixed/RandomTask
    Episode terminates if point goal is reached
    """

    def __init__(self, config):
        super(PointGoal, self).__init__(config)
        self.dist_tol = self.config.get("dist_tol", 0.5)

    def get_termination(self, task, env):
        """
        Return whether the episode should terminate.
        Terminate if point goal is reached (distance below threshold)

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """
        done = True if task.wanted_objects.sum() == 0 else False
        #done = task.wanted_object == 0
        success = done
        return done, success

class PointGoal_HRL(BaseTerminationCondition):
    """
    PointGoal used for PointNavFixed/RandomTask
    Episode terminates if point goal is reached
    """

    def __init__(self, config):
        super(PointGoal_HRL, self).__init__(config)
        self.dist_tol = self.config.get("dist_tol", 0.5)
        self.last_wanted_objects = None

    def get_termination(self, task, env):
        """
        Return whether the episode should terminate.
        Terminate if point goal is reached (distance below threshold)

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """
        if isinstance(self.last_wanted_objects,None):
            self.last_wanted_objects = task.wanted_objects_low_level
        if self.last_wanted_objects.sum() != task.wanted_objects_low_level.sum():
            done = True 
        else:
            done = False
        
        success = done
        return done, success
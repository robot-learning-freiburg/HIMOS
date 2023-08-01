from stable_baselines3.common.callbacks import BaseCallback
import os
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        print("REMAINING LR :",progress_remaining * initial_value)
        return progress_remaining * initial_value

    return func

class SaveModel(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str='./', verbose=1,model_id=1,hrl=False):
        super(SaveModel, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.curriculum_stage = 0
        self.save_path = os.path.join(log_dir, 'best_model_retrain_latest_{}'.format(self.curriculum_stage))
        self.model_id = model_id
        self.save_path_last = os.path.join(log_dir, 'last_model')


        self.best_mean_reward = -np.inf
        self.best_succ_rate = 0.0
        self.custom_env = None #self.model.env.envs[0]
        self.save_freq = 25000
        #self.buffer = None #self.model.rollout_buffer
        self.backward_episodes = 100
        self.current_stage_sum = 0
        self.use_hrl = hrl
        if self.use_hrl:
          self.save_path_low = os.path.join(log_dir, 'best_model_retrain_latest_low_{}'.format(self.curriculum_stage))
          self.save_path_last_low = os.path.join(log_dir, 'last_model_low')


    def _init_callback(self) -> None:
        # Create folder if needed
        #if self.save_path is not None:
        #    os.makedirs(self.save_path, exist_ok=True)

        self.custom_env = self.model.env
        #self.buffer = self.model.rollout_buffer
    def _on_step(self) -> bool:

        #check if model has failed status and reset buffer accordingly
        #if(self.custom_env.failed_status):
        #  self.model.n_steps -= self.custom_env.step_counter
        #  self.buffer.pos -= self.custom_env.step_counter
        #  print("removed current episode and set back counter's")
        #  self.env.reset()
        #check model checkpoint
        
        

                
        #print("Iteration",self.n_calls)
        if (self.n_calls+1) % self.check_freq == 0:
          """
          if self.current_stage_sum < self.custom_env.stages.sum():
            print("----->NEW STAGE HAS BEEN OBSERVED IN SAVE CALLBACK<------")
            self.best_mean_reward = 0
            
            self.current_stage_sum = self.custom_env.stages.sum()
            self.curriculum_stage += 1
            self.save_path = os.path.join(self.log_dir, 'best_model_retrain_latest_{}'.format(self.curriculum_stage))
          """
          """
          if(self.backward_episodes < 100):
            self.backward_episodes += 4
            
          if(self.custom_env.task.curriculum_stage != self.curriculum_stage):
            self.curriculum_stage = self.custom_env.task.curriculum_stage
            self.best_mean_reward = -np.inf
            self.save_path = os.path.join(self.log_dir, 'best_model_retrain_latest_{}'.format(self.curriculum_stage))
            self.backward_episodes = 4
            return 

          """
          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-self.backward_episodes:])
              


              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")


              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:

                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}.zip")
                  self.model.save(self.save_path,exclude=['low_level','ll_tensoboard_callback'])
                  if self.use_hrl:
                    self.model.low_level.save(self.save_path_low)

              print(f"Saving new LATEST model to {self.save_path}.zip")
              self.model.save(self.save_path_last,exclude=['low_level','ll_tensoboard_callback'])
              if self.use_hrl:
                self.model.low_level.save(self.save_path_last_low)
              
        
          """
          mean_succ = np.mean(self.custom_env.succ_rate)
          if(mean_succ > self.best_succ_rate):
            self.best_succ_rate = mean_succ
            print(f"Saving new best succ_model to {self.save_path_succ}.zip")
            self.model.save(self.save_path_succ)
          """

        """
        if (self.n_calls+1) % self.save_freq == 0:
            p = os.path.join(self.log_dir, 'model_retrain_2_{}'.format(self.n_calls))
            print(f"Saving model to {p}.zip")
            self.model.save(p)
        """

        return True
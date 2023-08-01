import os
from typing import Callable

import igibson
from src.igibson.envrionments.env import Env
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from src.SB3.save_model_callback import SaveModel, linear_schedule
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
import numpy as np
import cv2
import yaml
from hrl_models import CustomExtractorLL, CustomExtractorHL
try:
    import gym
    import torch as th
    import torch.nn as nn

    from src.exploration_policy.ppo_mod_disc import PPO as PPO_LL


    from src.SB3.ppo import PPO
    
    from src.highlevel_policy.general_policy import GEN_POLICY

    from src.highlevel_policy.vec_monitor_MOD import VecMonitor
    from src.highlevel_policy.subproc_vec_env_HRL import SubprocVecEnv
    

	
except ModuleNotFoundError:
    print("stable-baselines3 is not installed. You would need to do: pip install stable-baselines3")
    exit(1)


"""
Example training code using stable-baselines3 PPO for PointNav task.
"""
class SummaryWriterCallback(BaseCallback):

    def _on_training_start(self):
        self._log_freq = 1000  # log every 1000 calls

        output_formats = self.logger.output_formats
        self.low_level_pol = self.model.low_level
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:

        
        self.tb_formatter.writer.add_scalar("rollout/ep_rew_mean_low", safe_mean([ep_info["r"] for ep_info in self.low_level_pol.ep_info_buffer]),self.low_level_pol.num_timesteps)
        self.tb_formatter.writer.add_scalar("rollout/ep_len_mean_low", safe_mean([ep_info["l"] for ep_info in self.low_level_pol.ep_info_buffer]),self.low_level_pol.num_timesteps)
        self.tb_formatter.writer.flush()




def main():
    config_file = "config_train.yaml"
    tensorboard_log_dir = "log_dir"

    model_log_dir = ""  
    for i in range(10000000):
        model_log_dir = './model/{}/'.format(i)
        if(os.path.exists(model_log_dir)):
            continue
        else:
            break
    os.makedirs(model_log_dir, exist_ok=True)

    
    num_cpu = 32
    train_set = ['Merom_0_int', 'Benevolence_0_int', 'Pomaria_0_int', 'Wainscott_1_int', 'Rs_int', 'Ihlen_0_int',
                 'Beechwood_1_int', 'Ihlen_1_int',\
                  'Merom_0_int', 'Benevolence_0_int', 'Pomaria_0_int', 'Wainscott_1_int', 'Rs_int', 'Ihlen_0_int',
                 'Beechwood_1_int', 'Ihlen_1_int',\
                 'Merom_0_int', 'Wainscott_1_int', 'Pomaria_0_int', 'Wainscott_1_int', 'Wainscott_1_int', 'Ihlen_0_int',
                 'Beechwood_1_int', 'Ihlen_1_int',\
                 'Beechwood_1_int','Wainscott_1_int', 'Pomaria_0_int','Beechwood_1_int', 'Wainscott_1_int', 'Ihlen_0_int',
                 'Beechwood_1_int', 'Ihlen_1_int',\
                 'Beechwood_1_int','Wainscott_1_int', 'Wainscott_1_int','Wainscott_1_int', 'Wainscott_1_int', 'Wainscott_1_int',
                 'Ihlen_0_int', 'Ihlen_1_int']
    
    config_filename = os.path.join('./', 'config_train.yaml')
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
   
    num_discrete_actions = None
    if config_data.get("add_frontier_exploration",False):
        if config_data.get("add_exploration_policy", False):
            num_discrete_actions = 12
           
        else:
            num_discrete_actions = 11
          
    else:
        if config_data.get("add_exploration_policy", False):
            num_discrete_actions = 11
         
        else:
            num_discrete_actions = 10
          

    def make_env(rank: int, seed: int = 0, data_set=[]) -> Callable:
        def _init() -> Env:

            env = Env(config_filename=config_filename, scene_id = train_set[rank],mode="headless", use_pb_gui=False)

            env.seed(seed + rank)
            return env

        

        set_random_seed(seed)
        return _init

    
    

   
    all_envs = SubprocVecEnv([make_env(i, data_set=train_set) for i in range(num_cpu)],num_discrete_actions=num_discrete_actions)

    all_envs = VecMonitor(all_envs,filename=model_log_dir)
   
    policy_kwargs_LL = dict(
        features_extractor_class=CustomExtractorLL
    )
    policy_kwargs_HL = dict(
        features_extractor_class=CustomExtractorHL
    )
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    
   
    n_steps = 2048
    aux_bin_number = 12
    task_obs = all_envs.observation_space['task_obs'].shape[0] - aux_bin_number
    model_ll_pol = PPO_LL("MultiInputPolicy", all_envs, verbose=0,batch_size=2,n_steps=2,tensorboard_log=tensorboard_log_dir,device='auto', policy_kwargs=policy_kwargs_LL,aux_pred_dim=aux_bin_number,proprio_dim=task_obs,cut_out_aux_head=aux_bin_number)

    model_ll_pol.set_parameters("checkpoints/HIMOS_EP/last_model",exact_match=False) 
   
    
    all_envs.action_space = gym.spaces.Discrete(num_discrete_actions)

    
    if config_data.get("corrected_discounting", False):
        corrected_discounting = 0.998565
    else:
        corrected_discounting = 0.99
   
    model_hl_pol = PPO("MultiInputPolicy",all_envs,ent_coef=0.005,batch_size=128,gae_lambda=0.95,n_steps=n_steps,gamma=corrected_discounting,clip_range=0.1,n_epochs=4,learning_rate=0.0005, verbose=1,\
    tensorboard_log=tensorboard_log_dir, policy_kwargs=policy_kwargs_HL,config_data=config_data)
    
    model = GEN_POLICY(model_hl_pol,model_ll_pol,all_envs,config=config_data,num_envs=num_cpu)

    save_model_callback = SaveModel(check_freq=n_steps, log_dir=model_log_dir,hrl=False)

    model.learn(11500000,callback=[save_model_callback])
if __name__ == "__main__":
    main()
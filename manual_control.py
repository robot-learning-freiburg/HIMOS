import os
from typing import Callable

import igibson
from src.igibson.envrionments.env import Env
from src.SB3.save_model_callback import SaveModel
import numpy as np
import cv2
#from torchvision import models
#from VisTranNet import ViT
from hrl_models import CustomExtractorLL, CustomExtractorHL
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
import gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.utils import set_random_seed
from src.SB3.ppo import PPO
from stable_baselines3.common.monitor import Monitor
from src.highlevel_policy.general_policy import GEN_POLICY
from src.exploration_policy.ppo_mod_disc import PPO as PPO_LL
import yaml
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

from stable_baselines3.common.monitor import Monitor

    
from stable_baselines3.common import utils
    


def create_policy():
    config_filename = os.path.join('./', 'config.yaml')
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    
    env = Env(config_filename=config_filename, physics_timestep=1.0/120, action_timestep=1.0 / 10.0,mode="gui_interactive", use_pb_gui=True)#, rendering_settings=settings)
    

    policy_kwargs_LL = dict(
        features_extractor_class=CustomExtractorLL
    )
    policy_kwargs_HL = dict(
        features_extractor_class=CustomExtractorHL
    )

    aux_bin_number= 12
    task_obs = env.observation_space['task_obs'].shape[0] -aux_bin_number
    #845
    model_ll_pol = PPO_LL("MultiInputPolicy", env, verbose=0,batch_size=2,n_steps=2,device="cpu", policy_kwargs=policy_kwargs_LL,aux_pred_dim=aux_bin_number,proprio_dim=task_obs,cut_out_aux_head=aux_bin_number)
    

    model_ll_pol.set_parameters("checkpoints/LL_Fetch_7_81%/last_model",
                                exact_match=False)  

    
    if config_data.get("add_frontier_exploration", False):
        if config_data.get("add_exploration_policy", False):
            
            env.action_space = gym.spaces.Discrete(12)
        else:
            
            env.action_space = gym.spaces.Discrete(11)
    else:
        if config_data.get("add_exploration_policy", False):
            
            env.action_space = gym.spaces.Discrete(11)
        else:
            
            env.action_space = gym.spaces.Discrete(10)
    
    model_hl_pol = PPO("MultiInputPolicy",env, verbose=1, n_steps=2,batch_size=2,policy_kwargs=policy_kwargs_HL,config_data=config_data)#,tensorboard_low_level_callback=ll_tensorboard_callback)
    

    
    model = GEN_POLICY(model_hl_pol,model_ll_pol,env,config=config_data,num_envs=1)

    return model,env

def main():

    set_random_seed(5)
    
    model,env = create_policy()
    
    for ep in range(300):

        obs = env.reset()

        position_1 = [-2.59958985 , 2.47672702 , 0.0057725 ]

        
        while True:

            
            hl_ac = 0
            print("HL:")
            ac = input()

            if ac == "0":
                hl_ac = 0
            elif ac == "1":
                hl_ac = 1
            elif ac == "2":
                hl_ac = 2
            elif ac == "3":
                hl_ac = 3
            elif ac == "4":
                hl_ac = 4
            elif ac == "5":
                hl_ac = 5
            elif ac == "6":
                hl_ac = 6
            elif ac == "7":
                hl_ac = 7
            elif ac == "8":
                hl_ac = 8
            elif ac == "9":
                hl_ac = 9
            elif ac == "10":
                hl_ac = 10
            elif ac == "11":
                hl_ac = 11
            else:
                hl_ac = 0
            print("LL")
            ac = input()
            
            if(ac == "w"):
                
                action = np.array([1.0,0.0])#,0.0,0.0,0.0]
            elif(ac == "s"):
                
                action = np.array([-1.0,0.0])#,0.0,0.0,0.0]
            elif(ac == "a"):
                action = np.array([0.0,1.0])#,-1.0,0.0,0.0]
            elif(ac=="d"):
                action = np.array([0.0,-1.0])#,1.0,0.0,0.0]
            elif(ac=="q"):
                action = np.array([1.0,-1.0])#,1.0,0.0,0.0]
            elif(ac=="e"):
                action = np.array([1.0,1.0])#,1.0,0.0,0.0]

            elif(ac=="b"):
                break
            else:
                action = np.array([0.0,0.0])

            new_obs, rewards, dones, info =  env.step({"action":action,"hl_action":hl_ac,"aux_angle":np.array([0.0]*12)})
            ll_action = model.predict(new_obs,[0])

            print("Valid Actions:",new_obs['valid_actions'][1:-1])
            print("WANTED OBJECTS: ",new_obs['task_obs'][-6::])
            
            print(f"Rew. received: {rewards} and discount: {info['discount_length']}")


            if(dones):
                break

if __name__ == "__main__":
    main()






    
    
    

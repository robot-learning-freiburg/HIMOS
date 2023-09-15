import os
from typing import Callable
import numpy as np
import igibson
from src.igibson.envrionments.env import Env
from src.SB3.save_model_callback import SaveModel

from hrl_models import CustomExtractorLL, CustomExtractorHL
import torch
import gym

import gc
import yaml


from src.highlevel_policy.general_policy import GEN_POLICY

from src.SB3.ppo import PPO

from src.exploration_policy.ppo_mod_disc import PPO as PPO_LL

from src.highlevel_policy.subproc_vec_env_HRL import SubprocVecEnv

from stable_baselines3.common.utils import set_random_seed
from baselines.baseline1 import greedy_baseline

import random


def setup(scene_id, objects, method):
    config_filename = os.path.join('./', 'config_eval.yaml')
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    
    #to see visual interface of iGibson via PyBullet: use mode="gui_interactive" and  use_pb_gui=True
    env = Env(config_filename=config_filename, scene_id=scene_id, objects_find=objects, method=method,
              physics_timestep=1.0/120, action_timestep=1.0 / 10.0, mode="gui_non_interactive", use_pb_gui=False)

    policy_kwargs_LL = dict(
        features_extractor_class=CustomExtractorLL
    )
    policy_kwargs_HL = dict(
        features_extractor_class=CustomExtractorHL
    )

    aux_bin_number = 12
    task_obs = env.observation_space['task_obs'].shape[0] - aux_bin_number

    model_ll_pol = PPO_LL("MultiInputPolicy", env, verbose=0, batch_size=2, n_steps=2,
                          policy_kwargs=policy_kwargs_LL, aux_pred_dim=aux_bin_number, proprio_dim=task_obs, cut_out_aux_head=aux_bin_number)
    model_ll_pol.set_parameters("checkpoints/HIMOS_EP/last_model",
                                exact_match=False)  # previous checkpoint, used until 13.11.22


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


    exploration_policy_steps = config_data.get("exploration_policy_steps", 4)
    model_hl_pol = PPO("MultiInputPolicy", env, verbose=1, n_steps=2, batch_size=2,
                       policy_kwargs=policy_kwargs_HL, config_data=config_data)  


    model_hl_pol.set_parameters("checkpoints/HIMOS_HLP/seed_2/last_model_3",
                                exact_match=False) 
    
    model = GEN_POLICY(model_hl_pol, model_ll_pol, env, config=config_data, num_envs=1)

    return env, model_ll_pol, model_hl_pol, model, exploration_policy_steps


def set_determinism_eval(seed=0):
    set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def copy_changed_files():
    import shutil
    from pathlib import Path
    shutil.copyfile("requirements/fetch_gripper.urdf", "/opt/iGibson/igibson/data/assets/models/fetch/fetch_gripper.urdf")
    for file in Path("requirements").glob("**/*.png"):
        shutil.copyfile(file, f"/opt/iGibson/igibson/data/ig_dataset/scenes/{file.parent.name}/layout/{file.name}")

def main():
    copy_changed_files()
        
    
    scenes_counter = 8
    scenes_succ = {'Merom_0_int': [[] for i in range(6)],'Benevolence_0_int': [[] for i in range(6)],  'Pomaria_0_int': [[] for i in range(6)], 'Wainscott_1_int': [[] for i in range(6)],'Rs_int': [[] for i in range(6)],'Ihlen_0_int': [[] for i in range(6)], 'Beechwood_1_int': [[] for i in range(6)], 'Ihlen_1_int': [[] for i in range(6)]}
    scenes_spl = {'Merom_0_int': [[] for i in range(6)],'Benevolence_0_int': [[] for i in range(6)],  'Pomaria_0_int': [[] for i in range(6)], 'Wainscott_1_int': [[] for i in range(6)],'Rs_int': [[] for i in range(6)],'Ihlen_0_int': [[] for i in range(6)], 'Beechwood_1_int': [[] for i in range(6)], 'Ihlen_1_int': [[] for i in range(6)]}
    scenes_steps_taken_succ = {'Merom_0_int': [[] for i in range(6)],'Benevolence_0_int': [[] for i in range(6)],  'Pomaria_0_int': [[] for i in range(6)], 'Wainscott_1_int': [[] for i in range(6)],'Rs_int': [[] for i in range(6)],'Ihlen_0_int': [[] for i in range(6)], 'Beechwood_1_int': [[] for i in range(6)], 'Ihlen_1_int': [[] for i in range(6)]}
    scenes_steps_taken_no_succ = {'Merom_0_int': [[] for i in range(6)],'Benevolence_0_int': [[] for i in range(6)],  'Pomaria_0_int': [[] for i in range(6)], 'Wainscott_1_int': [[] for i in range(6)],'Rs_int': [[] for i in range(6)],'Ihlen_0_int': [[] for i in range(6)], 'Beechwood_1_int': [[] for i in range(6)], 'Ihlen_1_int': [[] for i in range(6)]}
    scenes_steps_general = {'Merom_0_int': [[] for i in range(6)],'Benevolence_0_int': [[] for i in range(6)],  'Pomaria_0_int': [[] for i in range(6)], 'Wainscott_1_int': [[] for i in range(6)],'Rs_int': [[] for i in range(6)],'Ihlen_0_int': [[] for i in range(6)], 'Beechwood_1_int': [[] for i in range(6)], 'Ihlen_1_int': [[] for i in range(6)]}
    test_scenes = ['Merom_0_int', 'Benevolence_0_int', 'Pomaria_0_int', 'Wainscott_1_int', 'Rs_int', 'Ihlen_0_int','Beechwood_1_int', 'Ihlen_1_int']
    """

    scenes_counter = 7
    scenes_succ = {'Pomaria_2_int': [[] for i in range(6)], 'Benevolence_2_int': [[] for i in range(6)],  'Benevolence_1_int': [[] for i in range(6)],  # noqa: E501
                   'Wainscott_0_int': [[] for i in range(6)], 'Beechwood_0_int': [[] for i in range(6)], 'Merom_1_int': [[] for i in range(6)], 'Pomaria_1_int': [[] for i in range(6)]}
    scenes_spl = {'Benevolence_1_int': [[] for i in range(6)], 'Pomaria_2_int': [[] for i in range(6)], 'Benevolence_2_int': [[] for i in range(6)],
                  'Wainscott_0_int': [[] for i in range(6)], 'Beechwood_0_int': [[] for i in range(6)], 'Pomaria_1_int': [[] for i in range(6)], 'Merom_1_int': [[] for i in range(6)]}
    scenes_steps_taken_succ = {'Benevolence_1_int': [[] for i in range(6)], 'Pomaria_2_int': [[] for i in range(6)], 'Benevolence_2_int': [
        []]*6, 'Wainscott_0_int': [[] for i in range(6)], 'Beechwood_0_int': [[] for i in range(6)], 'Pomaria_1_int': [[] for i in range(6)], 'Merom_1_int': [[] for i in range(6)]}
    scenes_steps_taken_no_succ = {'Benevolence_1_int': [[] for i in range(6)], 'Pomaria_2_int': [[] for i in range(6)], 'Benevolence_2_int': [
        []]*6, 'Wainscott_0_int': [[] for i in range(6)], 'Beechwood_0_int': [[] for i in range(6)], 'Pomaria_1_int': [[] for i in range(6)], 'Merom_1_int': [[] for i in range(6)]}
    scenes_steps_general = {'Benevolence_1_int': [[] for i in range(6)], 'Pomaria_2_int': [[] for i in range(6)], 'Benevolence_2_int': [
        []]*6, 'Wainscott_0_int': [[] for i in range(6)], 'Beechwood_0_int': [[] for i in range(6)], 'Pomaria_1_int': [[] for i in range(6)], 'Merom_1_int': [[] for i in range(6)]}
    test_scenes = ['Pomaria_2_int', 'Benevolence_2_int', 'Benevolence_1_int',
                   'Wainscott_0_int', 'Beechwood_0_int', 'Merom_1_int', 'Pomaria_1_int']
    """
    SPL_sum = []
    SR_sum = []

    scene_counter = 0

    baseline_greedy = greedy_baseline()
    
    method_eval = "greedy" #either greedy or policy
    method = f"HIMOS_eval_{method_eval}" #arbitary name
    seed = 22  # 22,42,64
    det_policy = False
    how_many_eps_per_sing_task = 25 
    objects_find = 0
    objects_find_max = 7
    wrong_commands = []
    ep_rew = 0
    discount_length_mean = []
    
    if not os.path.exists('eval_results'):
        os.makedirs('eval_results')
    assert not os.path.exists(f'eval_results/{method}_seed{seed}_succ.txt'), "File already exists"
    config_filename = os.path.join('./', 'config_eval.yaml')
    import shutil
    shutil.copyfile(config_filename, f'eval_results/{method}_seed{seed}_config.yaml')
    
    with open(f'eval_results/{method}_seed{seed}_succ.txt', 'w') as f:
        f.write('')
        f.close()

    with open(f'eval_results/{method}_seed{seed}_spl.txt', 'w') as f:
        f.write('')
        f.close()

    with open(f'eval_results/{method}_seed{seed}_steps.txt', 'w') as f:
        f.write('')
        f.close()

    p_dist_ex_fr = [[], []]
    env, model_ll_pol, model_hl_pol, model = None, None, None, None
    for ep in range(6000):
        print("#############################################")
        print(f"Starting episode {ep}")
        print("#############################################")
        acc_rew = []
        
        if ep % how_many_eps_per_sing_task == 0:

            sr_arr = np.array(scenes_succ[test_scenes[scene_counter]][objects_find-1])
            spl_arr = np.array(scenes_spl[test_scenes[scene_counter]][objects_find-1])

            steps_succ = np.array(scenes_steps_taken_succ[test_scenes[scene_counter]][objects_find-1])
            steps_no_succ = np.array(scenes_steps_taken_no_succ[test_scenes[scene_counter]][objects_find-1])
            steps_general = np.array(scenes_steps_general[test_scenes[scene_counter]][objects_find-1])

            SPL_sum.append(np.mean(spl_arr))
            SR_sum.append(np.mean(sr_arr))

            with open(f'eval_results/{method}_seed{seed}_succ.txt', 'a') as f:
                f.write(f'{np.mean(sr_arr)}+')
                f.close()

            with open(f'eval_results/{method}_seed{seed}_spl.txt', 'a') as f:
                f.write(f'{np.mean(spl_arr)}+')
                f.close()

            with open(f'eval_results/{method}_seed{seed}_steps.txt', 'a') as f:
                f.write(f'{np.mean(steps_succ)}+')
                f.close()

            print(f"-----Success-rate for scene {test_scenes[scene_counter]} and objects: {objects_find} : {np.mean(sr_arr)}")
            print(f"-----SPL for scene {test_scenes[scene_counter]} and objects: {objects_find} : {np.mean(spl_arr)}")

            print(f"-----Steps for scene Succ : {test_scenes[scene_counter]} and objects: {objects_find} : {np.mean(steps_succ)}")
            print(f"-----Steps for scene no-succ: {test_scenes[scene_counter]} and objects: {objects_find} : {np.mean(steps_no_succ)}")
            print(f"-----Steps for scene general: {test_scenes[scene_counter]} and objects: {objects_find} : {np.mean(steps_general)}")

           
            if objects_find == 0:
                env, model_ll_pol, model_hl_pol, model, exploration_policy_steps = setup(test_scenes[scene_counter], objects_find, method)

            objects_find += 1

            if objects_find == objects_find_max:
                scene_counter += 1
                
              
                del env, model_ll_pol, model_hl_pol, model

                gc.collect()
                torch.cuda.empty_cache()
                
                env, model_ll_pol, model_hl_pol, model, exploration_policy_steps = setup(test_scenes[scene_counter], objects_find, method)

                
                env.seed(seed)
                
                set_determinism_eval(seed)
                objects_find = 1

                with open(f'eval_results/{method}_seed{seed}_succ.txt', 'a') as f:
                    f.write('\n')
                with open(f'eval_results/{method}_seed{seed}_spl.txt', 'a') as f:
                    f.write('\n')
                with open(f'eval_results/{method}_seed{seed}_steps.txt', 'a') as f:
                    f.write('\n')

            env.task.num_tar_objects = objects_find

        obs = env.reset()
        
        baseline_greedy.reset()
        initial_geo_dist = env.task.initial_geodesic_length

        agent_geo_dist_taken = 0
        curr_position = env.robots[0].get_position()[:2]

        
        steps_counter = 0
        ep_rew = 0

        while True:
            if method_eval == "policy":
                hl_action, _ = model_hl_pol.predict(obs, deterministic=det_policy)
            else:
                hl_action = baseline_greedy.predict(env, obs)

            current_wrong_commands = env.wrong_command
            if hl_action == 0:
                num_ll_steps = exploration_policy_steps
            else:
                num_ll_steps = 1

            check_rew = 0
            discount_length = 0
            for ll_s in range(num_ll_steps):
                ll_action = model.predict(obs, [hl_action])
               
                steps_counter += 1

                new_obs, rewards, dones, info = env.step(ll_action)
                discount_length += info['discount_length']
                obs = new_obs

                ep_rew += rewards
                
                check_rew += rewards
                if dones:
                    break

            discount_length_mean.append(discount_length)

            
            new_position = env.robots[0].get_position()[:2]
            _, geodesic_dist = env.scene.get_shortest_path(env.task.floor_num, curr_position, new_position,
                                                           entire_path=False)
            curr_position = new_position
            agent_geo_dist_taken += geodesic_dist

            
            if(dones):
                
                scenes_steps_general[test_scenes[scene_counter]][objects_find-1].append(env.current_step)
                
                wrong_commands.append(current_wrong_commands)
                if(info['success']):
                    scenes_succ[test_scenes[scene_counter]][objects_find-1].append(1)
                    scenes_spl[test_scenes[scene_counter]][objects_find -
                                                           1].append(initial_geo_dist / max(initial_geo_dist, agent_geo_dist_taken))
                    scenes_steps_taken_succ[test_scenes[scene_counter]][objects_find-1].append(env.current_step)
                    
                else:
                    scenes_succ[test_scenes[scene_counter]][objects_find-1].append(0)
                    scenes_spl[test_scenes[scene_counter]][objects_find-1].append(0)
                    scenes_steps_taken_no_succ[test_scenes[scene_counter]][objects_find-1].append(env.current_step)
                    
                break


if __name__ == "__main__":
    main()

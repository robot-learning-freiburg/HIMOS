import os
import numpy as np
import torch
import gc
import time
import ray
import shutil
import yaml

from baselines.baseline1 import greedy_baseline, SGoLAM_baseline
from evaluate import setup, set_determinism_eval, copy_changed_files


@ray.remote(num_cpus=12)
def evaluate_scene(scene_id: str, method: str, seed, objects_find_max: int, how_many_eps_per_sing_task: int, det_policy: bool, method_eval: str):
    scenes_succ = {scene_id: [[] for i in range(6)]}
    scenes_spl = {scene_id: [[] for i in range(6)]}
    scenes_steps_taken_succ = {scene_id: [[] for i in range(6)]}
    scenes_steps_general = {scene_id: [[] for i in range(6)]}
    scenes_steps_taken_no_succ = {scene_id: [[] for i in range(6)]}

    objects_find_unused = 0
    env, model_ll_pol, model_hl_pol, model, exploration_policy_steps = setup(scene_id, objects_find_unused, method)
    env.seed(seed)
    set_determinism_eval(seed)
    if method_eval == "greedy":
        baseline = greedy_baseline()
    elif method_eval == "sgolam":
        baseline = SGoLAM_baseline()
    elif method_eval == "policy":
        pass
    else:
        raise ValueError(f"Invalid method_eval {method_eval}")

    tot_scene_ep = 0
    for objects_find in range(1, objects_find_max):
        env.task.num_tar_objects = objects_find

        for _e in range(how_many_eps_per_sing_task):
            print(f"Starting episode {tot_scene_ep} in scene {scene_id}, {objects_find} objects, episode {_e + 1} of this task")
            obs = env.reset()
            
            baseline.reset()
            initial_geo_dist = env.task.initial_geodesic_length

            agent_geo_dist_taken = 0
            curr_position = env.robots[0].get_position()[:2]

            steps_counter = 0
            ep_rew = 0

            while True:
                if method_eval == "policy":
                    hl_action, _ = model_hl_pol.predict(obs, deterministic=det_policy)
                else:
                    hl_action = baseline.predict(env, obs)

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
                
                new_position = env.robots[0].get_position()[:2]
                _, geodesic_dist = env.scene.get_shortest_path(env.task.floor_num, curr_position, new_position,
                                                            entire_path=False)
                curr_position = new_position
                agent_geo_dist_taken += geodesic_dist
                
                if(dones):
                    scenes_steps_general[scene_id][objects_find-1].append(env.current_step)
                    if(info['success']):
                        scenes_succ[scene_id][objects_find-1].append(1)
                        scenes_spl[scene_id][objects_find-1].append(initial_geo_dist / max(initial_geo_dist, agent_geo_dist_taken))
                        scenes_steps_taken_succ[scene_id][objects_find-1].append(env.current_step)
                        
                    else:
                        scenes_succ[scene_id][objects_find-1].append(0)
                        scenes_spl[scene_id][objects_find-1].append(0)
                        scenes_steps_taken_no_succ[scene_id][objects_find-1].append(env.current_step)
                        
                    break
            tot_scene_ep += 1

        sr_arr = np.array(scenes_succ[scene_id][objects_find-1])
        spl_arr = np.array(scenes_spl[scene_id][objects_find-1])

        steps_succ = np.array(scenes_steps_taken_succ[scene_id][objects_find-1])
        steps_no_succ = np.array(scenes_steps_taken_no_succ[scene_id][objects_find-1])
        steps_general = np.array(scenes_steps_general[scene_id][objects_find-1])

        print(f"-----Success-rate for scene {scene_id} and objects: {objects_find} : {np.mean(sr_arr)}")
        print(f"-----SPL for scene {scene_id} and objects: {objects_find} : {np.mean(spl_arr)}")

        print(f"-----Steps for scene Succ : {scene_id} and objects: {objects_find} : {np.mean(steps_succ)}")
        print(f"-----Steps for scene no-succ: {scene_id} and objects: {objects_find} : {np.mean(steps_no_succ)}")
        print(f"-----Steps for scene general: {scene_id} and objects: {objects_find} : {np.mean(steps_general)}")
            
    del env, model_ll_pol, model_hl_pol, model
    gc.collect()
    torch.cuda.empty_cache()
        
    return scenes_succ, scenes_spl, scenes_steps_taken_succ, scenes_steps_general, scenes_steps_taken_no_succ


def main():
    copy_changed_files()
    
    ray.init(num_cpus=60)

    method_eval = "greedy"  # greedy / sgolam / policy
    scenes_set = "seen"
    seed = 22  # 22,42,64
    det_policy = False
    how_many_eps_per_sing_task = 25
    objects_find_max = 7

    config_filename = os.path.join('./', 'config_eval.yaml')
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    method = f"HIMOS_eval_{method_eval}_{config_data['frontier_selection']}_{config_data['exploration_policy_steps']}steps_{config_data['max_step']}maxsteps_strategyRandom_{scenes_set}"  #arbitary name


    if scenes_set == "seen":
        scenes_succ = {'Merom_0_int': [[] for i in range(6)],'Benevolence_0_int': [[] for i in range(6)],  'Pomaria_0_int': [[] for i in range(6)], 'Wainscott_1_int': [[] for i in range(6)],'Rs_int': [[] for i in range(6)],'Ihlen_0_int': [[] for i in range(6)], 'Beechwood_1_int': [[] for i in range(6)], 'Ihlen_1_int': [[] for i in range(6)]}
        scenes_spl = {'Merom_0_int': [[] for i in range(6)],'Benevolence_0_int': [[] for i in range(6)],  'Pomaria_0_int': [[] for i in range(6)], 'Wainscott_1_int': [[] for i in range(6)],'Rs_int': [[] for i in range(6)],'Ihlen_0_int': [[] for i in range(6)], 'Beechwood_1_int': [[] for i in range(6)], 'Ihlen_1_int': [[] for i in range(6)]}
        scenes_steps_taken_succ = {'Merom_0_int': [[] for i in range(6)],'Benevolence_0_int': [[] for i in range(6)],  'Pomaria_0_int': [[] for i in range(6)], 'Wainscott_1_int': [[] for i in range(6)],'Rs_int': [[] for i in range(6)],'Ihlen_0_int': [[] for i in range(6)], 'Beechwood_1_int': [[] for i in range(6)], 'Ihlen_1_int': [[] for i in range(6)]}
        scenes_steps_taken_no_succ = {'Merom_0_int': [[] for i in range(6)],'Benevolence_0_int': [[] for i in range(6)],  'Pomaria_0_int': [[] for i in range(6)], 'Wainscott_1_int': [[] for i in range(6)],'Rs_int': [[] for i in range(6)],'Ihlen_0_int': [[] for i in range(6)], 'Beechwood_1_int': [[] for i in range(6)], 'Ihlen_1_int': [[] for i in range(6)]}
        scenes_steps_general = {'Merom_0_int': [[] for i in range(6)],'Benevolence_0_int': [[] for i in range(6)],  'Pomaria_0_int': [[] for i in range(6)], 'Wainscott_1_int': [[] for i in range(6)],'Rs_int': [[] for i in range(6)],'Ihlen_0_int': [[] for i in range(6)], 'Beechwood_1_int': [[] for i in range(6)], 'Ihlen_1_int': [[] for i in range(6)]}
        test_scenes = ['Merom_0_int', 'Benevolence_0_int', 'Pomaria_0_int', 'Wainscott_1_int', 'Rs_int', 'Ihlen_0_int','Beechwood_1_int', 'Ihlen_1_int']
    elif scenes_set == "unseen":
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
        test_scenes = ['Pomaria_2_int', 'Benevolence_2_int', 'Benevolence_1_int', 'Wainscott_0_int', 'Beechwood_0_int', 'Merom_1_int', 'Pomaria_1_int']
    else:
        raise ValueError("Invalid scene set")

    if not os.path.exists('eval_results'):
        os.makedirs('eval_results')
    fpath = f'eval_results/{method}_seed{seed}_succ.txt'
    assert not os.path.exists(fpath) or os.path.getsize(fpath) == 0, "File already exists and is not empty"
    config_filename = os.path.join('./', 'config_eval.yaml')
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

    result_refs = []
    for i, scene_id in enumerate(test_scenes):
        result_ref = evaluate_scene.remote(scene_id=scene_id, 
                                            method=method, 
                                            seed=seed + 999*i, 
                                            objects_find_max=objects_find_max, 
                                            how_many_eps_per_sing_task=how_many_eps_per_sing_task, 
                                            det_policy=det_policy, 
                                            method_eval=method_eval)
        result_refs.append(result_ref)
        # time.sleep(10)

    for i, scene_id in enumerate(test_scenes):
        result = ray.get(result_refs[i])
        _scenes_succ, _scenes_spl, _scenes_steps_taken_succ, _scenes_steps_general, _scenes_steps_taken_no_succ = result
        scenes_succ.update(_scenes_succ)
        scenes_spl.update(_scenes_spl)
        scenes_steps_taken_succ.update(_scenes_steps_taken_succ)
        scenes_steps_general.update(_scenes_steps_general)
        scenes_steps_taken_no_succ.update(_scenes_steps_taken_no_succ)
        
        for objects_find in range(1, objects_find_max):
            sr_arr = np.array(scenes_succ[scene_id][objects_find-1])
            spl_arr = np.array(scenes_spl[scene_id][objects_find-1])

            steps_succ = np.array(scenes_steps_taken_succ[scene_id][objects_find-1])
            # steps_no_succ = np.array(scenes_steps_taken_no_succ[scene_id][objects_find-1])
            # steps_general = np.array(scenes_steps_general[scene_id][objects_find-1])            
            with open(f'eval_results/{method}_seed{seed}_succ.txt', 'a') as f:
                f.write(f'{np.mean(sr_arr)}+')
                f.close()

            with open(f'eval_results/{method}_seed{seed}_spl.txt', 'a') as f:
                f.write(f'{np.mean(spl_arr)}+')
                f.close()

            with open(f'eval_results/{method}_seed{seed}_steps.txt', 'a') as f:
                f.write(f'{np.mean(steps_succ)}+')
                f.close()
        
        
        with open(f'eval_results/{method}_seed{seed}_succ.txt', 'a') as f:
            f.write('\n')
        with open(f'eval_results/{method}_seed{seed}_spl.txt', 'a') as f:
            f.write('\n')
        with open(f'eval_results/{method}_seed{seed}_steps.txt', 'a') as f:
            f.write('\n')
                
    print("Done!")
    ray.shutdown()


if __name__ == "__main__":
    main()

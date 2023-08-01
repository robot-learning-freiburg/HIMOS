import pybullet as p
import numpy as np
from igibson.external.pybullet_tools.utils import set_base_values_with_z
from igibson.utils.utils import cartesian_to_polar
from igibson.utils.utils import restoreState
from igibson.utils.utils import l2_distance
from time import sleep
import skimage
from functools import reduce 
import cv2



def check_collision(body_a, body_b=None, link_a=None, fixed_body_ids=None,ignore_ids=[]):


    pts = p.getContactPoints(bodyA=body_a, linkIndexA=link_a)
   

    collisions = [x for x in pts if x[2] not in ignore_ids]

   
    # contactDistance < 0 means actual penetration
    pts = [elem for elem in pts if elem[8] < 0.0 and elem[2] not in ignore_ids]

    # only count collision with fixed body ids if provided
    if fixed_body_ids is not None:
        pts = [elem for elem in pts if elem[2] in fixed_body_ids]

    return len(pts) > 0

def close_joints_by_index(body_id,joint_id,mode="max",ignore_ids=[],slow=False,simulator=None):
    state_id = p.saveState()
    
    j_low, j_high = p.getJointInfo(body_id, joint_id)[8:10]
    j_type = p.getJointInfo(body_id, joint_id)[2]
    parent_idx = p.getJointInfo(body_id, joint_id)[-1]
    if slow:
        step_size = np.pi / 36.0 if j_type == p.JOINT_REVOLUTE else 0.05
    
        for j_pos in np.arange(j_high + step_size,0.06, step=-step_size):
            
            p.setJointMotorControl2(body_id, joint_id, p.POSITION_CONTROL,j_pos,-1)
            simulator.step()
    else:
        
        p.resetJointState(body_id, joint_id, 0.0)
    
    p.removeState(state_id)

def open_joints_by_index(body_id,joint_id,mode="max",ignore_ids=[],simulator=None):
    state_id = p.saveState()
    
    j_low, j_high = p.getJointInfo(body_id, joint_id)[8:10]
    j_type = p.getJointInfo(body_id, joint_id)[2]
    parent_idx = p.getJointInfo(body_id, joint_id)[-1]
   
       
    step_size = np.pi / 36.0 if j_type == p.JOINT_REVOLUTE else 0.05
    
    for j_pos in np.arange(0.0, j_high + step_size, step=step_size):

       
        p.setJointMotorControl2(body_id, joint_id, p.POSITION_CONTROL,j_pos,1)

        simulator.step()
        
        
        
        has_collision = check_collision(body_a=body_id, link_a=joint_id,ignore_ids=ignore_ids)

        restoreState(state_id)
        if has_collision:
            p.resetJointState(body_id, joint_id, j_pos)
            break

    

    
    p.removeState(state_id)
def get_all_openable_joints(body_id):
        
    body_joint_pairs = []
    for joint_id in range(p.getNumJoints(body_id)):
        # cache current physics state
        state_id = p.saveState()

        j_low, j_high = p.getJointInfo(body_id, joint_id)[8:10]
        j_type = p.getJointInfo(body_id, joint_id)[2]
        parent_idx = p.getJointInfo(body_id, joint_id)[-1]
        if j_type not in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            p.removeState(state_id)
            continue
        # this is the continuous joint
        if j_low >= j_high:
            p.removeState(state_id)
            continue
        # this is the 2nd degree joint, ignore for now
        if parent_idx != -1:
            p.removeState(state_id)
            continue

            

        body_joint_pairs.append((body_id, joint_id))
        # Remove cached state to avoid memory leak.
        p.removeState(state_id)

        
    return body_joint_pairs


def hacky_cabinet_trick(body_id,joint_ids):
    
    for j_id in joint_ids:
        p.changeDynamics(body_id,j_id[1], mass=9999999.0,lateralFriction=0.1,spinningFriction=0.1,rollingFriction=0.1,frictionAnchor=True)

    p.changeDynamics(body_id, -1, activationState=p.ACTIVATION_STATE_WAKE_UP)


def close_all_drawers(body_id,joint_ids,ignore_ids):
    
    for i,id_ in enumerate(joint_ids):
        close_joints_by_index(body_id,id_[1],mode="max",ignore_ids=ignore_ids)
        
               

    
def drive_to_cabinet(env,ind,ignore_id,set_to_finish_pos=False):
    tmp_physics_timestep = env.simulator.physics_timestep_num
    env.simulator.physics_timestep_num = 1
    
    source = env.robots[0].base_link.get_position()[:2]

    #currently the last position in target_pos_list is the cabinet. need to properly change it
    selected_microwave_pos = env.task.target_pos_list[ind].copy()

    offset_pos = env.task.pos_offset_agent_to_cabinets[str(env.task.target_pos_orient_list[ind])]
    
    selected_microwave_pos += offset_pos
    

    waypoints, _ = env.scene.get_shortest_path(env.task.floor_num, source, selected_microwave_pos[:2],
                                                entire_path=True)
    
    if env.short_horizon_planning:
        remaining_wps = len(waypoints)-env.num_wps_cut
        waypoints = waypoints[:env.num_wps_cut]
    else:
        remaining_wps = 0

    waypoint_yaw = env.robots[0].get_rpy()[2]
    
    prev_pos = [source[0], source[1], waypoint_yaw]
    stop_before = 6
    # skip two or three steps in order to not let the agent run into the opened door
    # append the first waypoints up the the 6th last and then skip a few points and add the last 2 again

    if len(waypoints) > 8:
        num_skips = 8
    else:
        num_skips = 5


    if env.animate:
        num_skips = 1
    
    driven_wps = 1
    for i in range(0, len(waypoints)): 

        point = waypoints[i]
        next_base_pos = env.mapping.world2map(np.array([point[0], point[1]]))
        env.global_map[int(next_base_pos[1]) - 2:int(next_base_pos[1]) + 2, int(next_base_pos[0]) - 2:int(next_base_pos[0]) + 2] = env.mapping.trace
       
        if len(waypoints) > (i + 1):
            nx_w = waypoints[i + 1]
            next_waypoint_local = env.task.global_to_local(env, [nx_w[0], nx_w[1], 0.5])[:2]
            next_waypoint_local_polar = np.array(cartesian_to_polar(next_waypoint_local[0], next_waypoint_local[1]))
            
            if env.animate:
                max_turn_angle = 0.1
                step_size = abs(int(np.round(next_waypoint_local_polar[1] / max_turn_angle)))
                if step_size > 1:
                    step_list = np.linspace(0.1,abs(next_waypoint_local_polar[1]),num=step_size)
                    for next_angle in step_list:
                        if next_waypoint_local_polar[1] > 0.0:
                            set_base_values_with_z(env.robots[0].get_body_ids()[0], [point[0], point[1], waypoint_yaw + next_angle],
                                       z=env.initial_pos_z_offset_2)
                        else:
                            set_base_values_with_z(env.robots[0].get_body_ids()[0], [point[0], point[1], waypoint_yaw - next_angle],
                                       z=env.initial_pos_z_offset_2)
                        sensors = env.get_sensor_obs()
                        ego1, ego2 = env.mapping.run_mapping(env, sensors, action=None)
                        cv2.imwrite('data/vid{}/coarse/{}_{}'.format(env.current_episode,env.global_counter,'.png'),ego2.astype(np.uint8))
                        cv2.imwrite('data/vid{}/fine/{}_{}'.format(env.current_episode,env.global_counter,'.png'),ego1.astype(np.uint8))
                        cv2.imwrite('data/vid{}/rgb/{}_{}'.format(env.current_episode,env.global_counter,'.png'),cv2.cvtColor(sensors['rgb']*255,cv2.COLOR_RGB2BGR))
                        env.global_counter += 1
            waypoint_yaw = waypoint_yaw + next_waypoint_local_polar[1]
            

        set_base_values_with_z(env.robots[0].get_body_ids()[0], [point[0], point[1], waypoint_yaw],
                               z=env.initial_pos_z_offset_2)

        driven_wps += 1
        if i % num_skips == 0:
            collision = env.run_sim_for_planner(ignore_id)    

            if collision :
                avoidance_succ,new_ind = skip_wp_avoid_collisions(env,driven_wps,waypoints)
                #sometimes, the agent bumps into cabinets and then the drawers fly around, making it impossible to check them out.
               
                if avoidance_succ :
                    i = new_ind
                else:
                    set_base_values_with_z(env.robots[0].get_body_ids()[0], prev_pos, z=env.initial_pos_z_offset_2)
                    # run one or maybe more sim steps. Avoid mapping fragments when the robot bumbs into something
                    # let the robot stabalize first and then continue mapping in .step() function later
                    env.simulator.step()
                    env.simulator.physics_timestep_num = int(tmp_physics_timestep)
                    return True,driven_wps,remaining_wps>0
                    
                
            else:
                env.simulator.sync()            
                
            prev_pos = [waypoints[i][0], waypoints[i][1], waypoint_yaw]
         
    env.simulator.physics_timestep_num = int(tmp_physics_timestep)
    return False,driven_wps,remaining_wps>0
    

def drive_to_selected_cracker(env,ind):
    
    tmp_physics_timestep = env.simulator.physics_timestep_num
    env.simulator.physics_timestep_num = 1

    source = env.robots[0].base_link.get_position()[:2]

    
    selected_cracker_pos = env.task.target_pos_list[3+ind]

    
    waypoints, _ = env.scene.get_shortest_path(env.task.floor_num, source, selected_cracker_pos[:2],
                                                entire_path=True)
    
    waypoint_yaw = env.robots[0].get_rpy()[2]
    prev_pos = [source[0], source[1], waypoint_yaw]
    stop_before = 6
    # skip two or three steps in order to not let the agent run into the opened door
    # append the first waypoints up the the 6th last and then skip a few points and add the last 2 again
    
    
    waypoints = waypoints[0:-5]

    if env.short_horizon_planning:
        remaining_wps = len(waypoints)-env.num_wps_cut
        waypoints = waypoints[:env.num_wps_cut]
    else:
        remaining_wps = 0


    if len(waypoints) > 8:
        num_skips = 8
    else:
        num_skips = 5

    if env.animate:
        num_skips = 1
    encountered_collision = False
    
    driven_wps = 1
    i = 0
    while i < len(waypoints):
   

        point = waypoints[i]
        next_base_pos = env.mapping.world2map(np.array([point[0], point[1]]))
        env.global_map[int(next_base_pos[1]) - 2:int(next_base_pos[1]) + 2, int(next_base_pos[0]) - 2:int(next_base_pos[0]) + 2] = env.mapping.trace
        
        # set the next yaw angle to the following waypoint in order to look porperly into the locomotion direction
        if len(waypoints) > (i + 1):
            nx_w = waypoints[i + 1]
            next_waypoint_local = env.task.global_to_local(env, [nx_w[0], nx_w[1], 0.5])[:2]
            next_waypoint_local_polar = np.array(cartesian_to_polar(next_waypoint_local[0], next_waypoint_local[1]))
            
            if env.animate:
                max_turn_angle = 0.1
                step_size = abs(int(np.round(next_waypoint_local_polar[1] / max_turn_angle)))
                if step_size > 1:
                    step_list = np.linspace(0.1,abs(next_waypoint_local_polar[1]),num=step_size)
                    for next_angle in step_list:
                        if next_waypoint_local_polar[1] > 0.0:
                            set_base_values_with_z(env.robots[0].get_body_ids()[0], [point[0], point[1], waypoint_yaw + next_angle],
                                       z=env.initial_pos_z_offset_2)
                        else:
                            set_base_values_with_z(env.robots[0].get_body_ids()[0], [point[0], point[1], waypoint_yaw - next_angle],
                                       z=env.initial_pos_z_offset_2)
                        sensors = env.get_sensor_obs()
                        ego1, ego2 = env.mapping.run_mapping(env, sensors, action=None)
                        cv2.imwrite('data/vid{}/coarse/{}_{}'.format(env.current_episode,env.global_counter,'.png'),ego2.astype(np.uint8))
                        cv2.imwrite('data/vid{}/fine/{}_{}'.format(env.current_episode,env.global_counter,'.png'),ego1.astype(np.uint8))
                        cv2.imwrite('data/vid{}/rgb/{}_{}'.format(env.current_episode,env.global_counter,'.png'),cv2.cvtColor(sensors['rgb']*255,cv2.COLOR_RGB2BGR))
                        env.global_counter += 1
            waypoint_yaw = waypoint_yaw + next_waypoint_local_polar[1]

        set_base_values_with_z(env.robots[0].get_body_ids()[0], [point[0], point[1], waypoint_yaw],
                               z=env.initial_pos_z_offset_2)
        driven_wps += 1

        if i % num_skips == 0:
            collision = env.run_sim_for_planner([])
            if collision:
                
                avoidance_succ,new_ind = skip_wp_avoid_collisions(env,driven_wps,waypoints)
                if avoidance_succ:
                    i = new_ind
                else:
                    set_base_values_with_z(env.robots[0].get_body_ids()[0], prev_pos, z=env.initial_pos_z_offset_2)
                    # run one or maybe more sim steps. Avoid mapping fragments when the robot bumbs into something
                    # let the robot stabalize first and then continue mapping in .step() function later
                    env.simulator.step()
                    encountered_collision = True
                    env.simulator.physics_timestep_num = int(tmp_physics_timestep)
                    return True,driven_wps,remaining_wps>0
                # break
            else:
                env.simulator.sync()
                
              

            prev_pos = [waypoints[i][0], waypoints[i][1], waypoint_yaw]
           
        i +=1 

    
       
    env.simulator.physics_timestep_num = int(tmp_physics_timestep)
    return False,driven_wps,remaining_wps>0
    

def drive_to_selected_door(ind,env,set_to_finish_pos=False):
    
    tmp_physics_timestep = env.simulator.physics_timestep_num
    env.simulator.physics_timestep_num = 1

    source = env.robots[0].base_link.get_position()[:2]

    key = env.task.door_index_to_key[ind]
    selected_door_pos = env.task.door_dict[key][1]

    offset_pos_axis = env.task.pos_offset_axis_agent_to_doors[str(env.task.door_dict[key][2])]
    
    waypoints, _ = env.scene.get_shortest_path(env.task.floor_num, source, selected_door_pos[:2],
                                                entire_path=True)
    

    point_comparison = None
    if len(waypoints) < 3:
        point_comparison = source
    else:
        point_comparison = waypoints[-3]

   
    which_side = None
    door_side = None
    
    if env.physically_simulated_openings:
        door_opens_towards = {"-2.0":0,"0.0":0,"-0.0":0,"2.0":1,"-3.0":1,"3.0":1,"1.0":0,"-1.0":0}
       

        if point_comparison[offset_pos_axis] < selected_door_pos[offset_pos_axis]:
            door_side = door_opens_towards[str(env.task.door_dict[key][2])]
            which_side = 0
            
            new_base = np.zeros(3)
            new_base[0] = selected_door_pos[0]
            new_base[1] = selected_door_pos[1]
            
            standard_offset = 0.86
            standard_offset_2 = 0.20
            if "special_offset" in env.task.offset_positions_door_knop[env.last_scene_id][key]:
                standard_offset = 0.75
                standard_offset_2 = 0.3
            elif "special_offset_2" in env.task.offset_positions_door_knop[env.last_scene_id][key]:
                standard_offset = 0.9
                standard_offset_2 = 0.3

            elif "special_offset_3" in env.task.offset_positions_door_knop[env.last_scene_id][key]:
                standard_offset = 0.9
                standard_offset_2 = 0.0

            elif "special_offset_4" in env.task.offset_positions_door_knop[env.last_scene_id][key]:
                standard_offset = 0.5
                standard_offset_2 = 0.5
            elif "special_offset_5" in env.task.offset_positions_door_knop[env.last_scene_id][key]:
                standard_offset = 0.55
                standard_offset_2 = 0.5
         

            if door_side != which_side:
                new_base[offset_pos_axis] -= 0.5
            else:
                new_base[offset_pos_axis] -= standard_offset

                if offset_pos_axis == 1:
                    new_base[0] -= standard_offset_2
                else:   
                    new_base[1] += standard_offset_2
        else:
            which_side = 1
            door_side = door_opens_towards[str(env.task.door_dict[key][2])]
            
            new_base = np.zeros(3)
            new_base[0] = selected_door_pos[0]
            new_base[1] = selected_door_pos[1]
       
            standard_offset = 0.86
            standard_offset_2 = 0.20
            if "special_offset" in env.task.offset_positions_door_knop[env.last_scene_id][key]:
           
                standard_offset = 0.75
                standard_offset_2 = 0.3

            elif "special_offset_2" in env.task.offset_positions_door_knop[env.last_scene_id][key]:
                standard_offset = 0.9
                standard_offset_2 = 0.3

            elif "special_offset_3" in env.task.offset_positions_door_knop[env.last_scene_id][key]:
                standard_offset = 0.9
                standard_offset_2 = 0.0

            elif "special_offset_4" in env.task.offset_positions_door_knop[env.last_scene_id][key]:
                standard_offset = 0.5
                standard_offset_2 = 0.5
            elif "special_offset_5" in env.task.offset_positions_door_knop[env.last_scene_id][key]:
                standard_offset = 0.75
                standard_offset_2 = 0.3
          

            if door_side != which_side:
                new_base[offset_pos_axis] += 0.5
            else:
                new_base[offset_pos_axis] += standard_offset
                if offset_pos_axis == 0:
                    new_base[1] -= standard_offset_2
                else:
                    new_base[0] += standard_offset_2
    else:
    
    
        if point_comparison[offset_pos_axis] < selected_door_pos[offset_pos_axis]:
            new_base = np.zeros(3)#tar_pos#.copy()
            new_base[0] = selected_door_pos[0]
            new_base[1] = selected_door_pos[1]
        
            new_base[offset_pos_axis] -= 0.8
        else:
            new_base = np.zeros(3)
            new_base[0] = selected_door_pos[0]
            new_base[1] = selected_door_pos[1]
       
            new_base[offset_pos_axis] += 0.8

     
    
    waypoints, _ = env.scene.get_shortest_path(env.task.floor_num, source, new_base[:2],
                                                entire_path=True)
    
    if env.short_horizon_planning:
        remaining_wps = len(waypoints)-env.num_wps_cut
        waypoints = waypoints[:env.num_wps_cut]
    else:
        remaining_wps = 0

    waypoint_yaw = env.robots[0].get_rpy()[2]
    prev_pos = [source[0], source[1], waypoint_yaw]
    stop_before = 6
    if len(waypoints) > 8:
        num_skips = 8
    else:
        num_skips = 5
    
    if env.animate:
        num_skips = 1

    driven_wps = 1
    i = 0
    while i < len(waypoints):
  

        point = waypoints[i]
        next_base_pos = env.mapping.world2map(np.array([point[0], point[1]]))
        env.global_map[int(next_base_pos[1]) - 2:int(next_base_pos[1]) + 2, int(next_base_pos[0]) - 2:int(next_base_pos[0]) + 2] = env.mapping.trace
        


      
        if len(waypoints) > (i + 1):
            nx_w = waypoints[i + 1]
            next_waypoint_local = env.task.global_to_local(env, [nx_w[0], nx_w[1], 0.5])[:2]
            next_waypoint_local_polar = np.array(cartesian_to_polar(next_waypoint_local[0], next_waypoint_local[1]))

            if env.animate:
                max_turn_angle = 0.1
                step_size = abs(int(np.round(next_waypoint_local_polar[1] / max_turn_angle)))
                if step_size > 1:
                    step_list = np.linspace(0.1,abs(next_waypoint_local_polar[1]),num=step_size)
                    for next_angle in step_list:
                        if next_waypoint_local_polar[1] > 0.0:
                            set_base_values_with_z(env.robots[0].get_body_ids()[0], [point[0], point[1], waypoint_yaw + next_angle],
                                       z=env.initial_pos_z_offset_2)
                        else:
                            set_base_values_with_z(env.robots[0].get_body_ids()[0], [point[0], point[1], waypoint_yaw - next_angle],
                                       z=env.initial_pos_z_offset_2)
                        sensors = env.get_sensor_obs()
                        ego1, ego2 = env.mapping.run_mapping(env, sensors, action=None)
                        cv2.imwrite('data/vid{}/coarse/{}_{}'.format(env.current_episode,env.global_counter,'.png'),ego2.astype(np.uint8))
                        cv2.imwrite('data/vid{}/fine/{}_{}'.format(env.current_episode,env.global_counter,'.png'),ego1.astype(np.uint8))
                        cv2.imwrite('data/vid{}/rgb/{}_{}'.format(env.current_episode,env.global_counter,'.png'),cv2.cvtColor(sensors['rgb']*255,cv2.COLOR_RGB2BGR))
                        env.global_counter += 1

            waypoint_yaw = waypoint_yaw + next_waypoint_local_polar[1]

                
        set_base_values_with_z(env.robots[0].get_body_ids()[0], [point[0], point[1], waypoint_yaw],
                               z=env.initial_pos_z_offset_2)

        driven_wps += 1
        if i % num_skips == 0:
            collision = env.run_sim_for_planner([])
            if collision:
                
                avoidance_succ,new_ind = skip_wp_avoid_collisions(env,driven_wps,waypoints)

                
                if avoidance_succ:
                    i = new_ind
                    
                else:
                    set_base_values_with_z(env.robots[0].get_body_ids()[0], prev_pos, z=env.initial_pos_z_offset_2)
                    # run one or maybe more sim steps. Avoid mapping fragments when the robot bumbs into something
                    # let the robot stabalize first and then continue mapping in .step() function later
                    env.simulator.step()
                    encountered_collision = True
                    env.simulator.physics_timestep_num = int(tmp_physics_timestep)
                    return True,driven_wps,remaining_wps>0,which_side

            else:
                env.simulator.sync()    
          

            prev_pos = [waypoints[i][0], waypoints[i][1], waypoint_yaw]
        i+=1
       
    env.simulator.physics_timestep_num = int(tmp_physics_timestep)
    return False,driven_wps,remaining_wps>0,which_side
    

def skip_wp_avoid_collisions(env,driven_wps,wanted_wp):
    
    num_skips = 1
    waypoint_yaw = env.robots[0].get_rpy()[2]
  
    recover_map = env.global_map.copy()
    for i in range(driven_wps, len(wanted_wp), num_skips):  
        
        point = wanted_wp[i]
        next_base_pos = env.mapping.world2map(np.array([point[0], point[1]]))

        # set the next yaw angle to the following waypoint in order to look porperly into the locomotion direction
        
        set_base_values_with_z(env.robots[0].get_body_ids()[0], [point[0], point[1], waypoint_yaw],
                               z=env.initial_pos_z_offset_2)

       
        
        env.global_map[int(next_base_pos[1]) - 2:int(next_base_pos[1]) + 2, int(next_base_pos[0]) - 2:int(next_base_pos[0]) + 2] = env.mapping.trace
        collision = env.run_sim_for_planner([])
        
        if collision:
            continue
        else:
            
            return True,i 

   
    env.global_map = recover_map
    return False, driven_wps

def drive_to_frontier_point(env,frontier_point):

    tmp_physics_timestep = env.simulator.physics_timestep_num
    env.simulator.physics_timestep_num = 1

    source = env.robots[0].base_link.get_position()[:2]
    
    waypoints, _ = env.scene.get_shortest_path(env.task.floor_num, source, frontier_point[:2],
                                                entire_path=True)

    if env.short_horizon_planning:
        remaining_wps = len(waypoints)-env.num_wps_cut
        waypoints = waypoints[:env.num_wps_cut]
    else:
        remaining_wps = 0


    waypoint_yaw = env.robots[0].get_rpy()[2]
    prev_pos = [source[0], source[1], waypoint_yaw]
    
    
    waypoints = waypoints[0:-3]
    
    
    if len(waypoints) > 8:
        num_skips = 8
    else:
        num_skips = 5


    if env.animate:
        num_skips = 1
        
    driven_wps = 1
    i = 0
    while i < len(waypoints):
   

        point = waypoints[i]
        next_base_pos = env.mapping.world2map(np.array([point[0], point[1]]))

        env.global_map[int(next_base_pos[1]) - 2:int(next_base_pos[1]) + 2, int(next_base_pos[0]) - 2:int(next_base_pos[0]) + 2] = env.mapping.trace

       

        # set the next yaw angle to the following waypoint in order to look porperly into the locomotion direction
        if len(waypoints) > (i + 1):
            nx_w = waypoints[i + 1]
            next_waypoint_local = env.task.global_to_local(env, [nx_w[0], nx_w[1], 0.5])[:2]
            next_waypoint_local_polar = np.array(cartesian_to_polar(next_waypoint_local[0], next_waypoint_local[1]))
            if env.animate:
                max_turn_angle = 0.1
                step_size = abs(int(np.round(next_waypoint_local_polar[1] / max_turn_angle)))
                if step_size > 1:
                    step_list = np.linspace(0.1,abs(next_waypoint_local_polar[1]),num=step_size)
                    for next_angle in step_list:
                        if next_waypoint_local_polar[1] > 0.0:
                            set_base_values_with_z(env.robots[0].get_body_ids()[0], [point[0], point[1], waypoint_yaw + next_angle],
                                       z=env.initial_pos_z_offset_2)
                        else:
                            set_base_values_with_z(env.robots[0].get_body_ids()[0], [point[0], point[1], waypoint_yaw - next_angle],
                                       z=env.initial_pos_z_offset_2)
                        sensors = env.get_sensor_obs()
                        ego1, ego2 = env.mapping.run_mapping(env, sensors, action=None)
                        cv2.imwrite('data/vid{}/coarse/{}_{}'.format(env.current_episode,env.global_counter,'.png'),ego2.astype(np.uint8))
                        cv2.imwrite('data/vid{}/fine/{}_{}'.format(env.current_episode,env.global_counter,'.png'),ego1.astype(np.uint8))
                        cv2.imwrite('data/vid{}/rgb/{}_{}'.format(env.current_episode,env.global_counter,'.png'),cv2.cvtColor(sensors['rgb']*255,cv2.COLOR_RGB2BGR))
                        env.global_counter += 1

            waypoint_yaw = waypoint_yaw + next_waypoint_local_polar[1]

        set_base_values_with_z(env.robots[0].get_body_ids()[0], [point[0], point[1], waypoint_yaw],
                               z=env.initial_pos_z_offset_2)

        driven_wps += 1
        
        if i % num_skips == 0:
            collision = env.run_sim_for_planner([])

            if collision:
               
                avoidance_succ,new_ind = skip_wp_avoid_collisions(env,driven_wps,waypoints)

                
                if avoidance_succ:
                    i = new_ind
                else:
                    
                    set_base_values_with_z(env.robots[0].get_body_ids()[0], prev_pos, z=env.initial_pos_z_offset_2)
                    # run one or maybe more sim steps. Avoid mapping fragments when the robot bumbs into something
                    # let the robot stabalize first and then continue mapping in .step() function later
                    env.simulator.step()
                    env.simulator.physics_timestep_num = int(tmp_physics_timestep)
                    return True,driven_wps,remaining_wps>0
                # break
            else:
                env.simulator.sync()
                
              

            prev_pos = [waypoints[i][0], waypoints[i][1], waypoint_yaw]
            
        i += 1

    env.simulator.physics_timestep_num = int(tmp_physics_timestep)
    collision = env.run_sim_for_planner([])
    if collision:
        set_base_values_with_z(env.robots[0].get_body_ids()[0], prev_pos, z=env.initial_pos_z_offset_2)
        return True,driven_wps,remaining_wps>0

    
    return False,driven_wps,remaining_wps>0
   

def explore(sem_map,map_size_,agent_pos):
        
    
    occupancy_map = np.zeros((map_size_,map_size_,1))
    free_space = sem_map[:,:,0] == 255
    trace = sem_map[:,:,2] == 255
    walls = sem_map[:,:,1] == 255

    #mark cracker objects as free space as well, since in Ben0 the corridors will be blocked otherwise. Last one representes cracker found color
    crackers = reduce(np.logical_or, [sem_map[:,:,0] == 64, sem_map[:,:,0] == 32, sem_map[:,:,0] == 12, sem_map[:,:,0] == 249])
   

    #mask all stuff which has received a color and assign no free space "1.0" to it
    masks = reduce(np.logical_and, [sem_map[:,:,0] != 0 ,sem_map[:,:,1] != 0 ,sem_map[:,:,2] != 0])
    occupancy_map[masks] = 1.0

    #free space and trace are considered as free
    occupancy_map[free_space] = 0.5
    occupancy_map[trace] = 0.5
    occupancy_map[crackers] = 0.5

    occupancy_map[walls] = 1.0
    kernel = np.ones((5,5),np.uint8)
    occupancy_map = cv2.dilate(occupancy_map,kernel,iterations = 1)[:,:,np.newaxis]
    #-------------------------------------------------------------------------------------------------------------------
    explore_map = occupancy_map
    temp_occp = explore_map
    explored_area_map_t = np.ones((map_size_,map_size_,1))
    explored_area_map_t[temp_occp == 0.5] = 0.5 
    components_img = explored_area_map_t.reshape([map_size_, map_size_])
    
    components_img[int(agent_pos[1]) - 4: int(agent_pos[1]) + 4, int(agent_pos[0]) - 4: int(agent_pos[0]) + 4] = 0.5  # Make sure agent is on 'open'
    components_labels, num = skimage.morphology.label(components_img, connectivity = 2, background = 1, return_num = True)
    connected_idx = (components_labels == components_labels[int(agent_pos[1]),int(agent_pos[0])])
    _, counts = np.unique(components_labels, return_counts=True)
    largest_area_idx = np.argmax(counts[1:]) + 1
    largest_open = (components_labels == largest_area_idx)

    selem = skimage.morphology.disk(1)
    map_np = temp_occp.reshape([map_size_,map_size_])
    occupied_idx = (map_np == 1)
    unexp_idx = (map_np == 0)
    empty_idx = (map_np == 0.5)
    neighbor_unexp_idx = (skimage.filters.rank.minimum((map_np * 255).astype(np.uint8), selem) == 0)
    neighbor_occp_idx = (skimage.filters.rank.maximum((map_np * 255).astype(np.uint8),selem) == 255)
    frontier_idx = empty_idx & neighbor_unexp_idx & (~neighbor_occp_idx)

    valid_idx = frontier_idx & connected_idx
    

    cluster_img = np.zeros([map_size_, map_size_], dtype=np.uint8)
    cluster_img[valid_idx] = 1
    labels_cluster, num = skimage.measure.label(cluster_img, connectivity = 2, return_num = True)
    if cluster_img.sum() !=0:
        unique, counts = np.unique(labels_cluster, return_counts = True)
        
        largest_cluster_label = np.argmax(counts[1:])+1
        output_img = np.zeros([map_size_,map_size_])
        output_img[labels_cluster == largest_cluster_label] = 1
        output_img[labels_cluster != largest_cluster_label] = 0 
        final_idx = (output_img == 1)
       
        final_idx = final_idx
        x_np = np.where(final_idx == True)[0]
        y_np = np.where(final_idx == True)[1]

        x_mat = np.transpose(np.vstack([x_np]*np.size(x_np))) - x_np
        y_mat = np.transpose(np.vstack([y_np]*np.size(y_np))) - y_np
        sum_val = np.sum((x_mat**2 + y_mat**2)**(1/2),1)
        medoid = np.argmin(sum_val)

        goal_position = np.array([x_np[medoid], y_np[medoid]])#, device=device).long() 
        
    else:
        goal_position = None
        
    return goal_position

    
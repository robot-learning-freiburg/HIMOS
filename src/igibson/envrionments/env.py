from collections import OrderedDict

import cv2
import gym
import numpy as np
import pybullet as p
import yaml
from igibson.external.pybullet_tools.utils import (set_base_values_with_z,
                                                   stable_z_on_aabb)
from igibson.objects.visual_marker import VisualMarker
from igibson.render.mesh_renderer.mesh_renderer_settings import \
    MeshRendererSettings
from igibson.render.profiler import Profiler
from igibson.robots import REGISTERED_ROBOTS
from igibson.robots.robot_base import BaseRobot
from igibson.simulator import Simulator
from igibson.utils.constants import MAX_CLASS_COUNT
from igibson.utils.grasp_planning_utils import get_grasp_position_for_open
from igibson.utils.utils import (cartesian_to_polar, l2_distance, parse_config,
                                 quatToXYZW)
from scipy.stats import circvar
from transforms3d import euler
from transforms3d.euler import euler2quat

from igibson import object_states
from src.igibson.robot.fetch import Fetch_DD
# tmp
from src.igibson.robot.locobot import Locobot_DD
from src.igibson.scene.igibson_indoor_scene import InteractiveIndoorScene
from src.igibson.task.hrl_task import HRLTask
from src.utils.mapping_module import MappingModule
from src.utils.motion_planner import MotionPlanner
from src.utils.utils import (close_all_drawers, close_joints_by_index,
                             drive_to_cabinet, drive_to_frontier_point,
                             drive_to_selected_cracker, drive_to_selected_door,
                             explore, open_joints_by_index)


class Env(gym.Env):
    def __init__(self, config_filename, objects_find=None, scene_id=None, method="policy", mode="headless", action_timestep=1 / 10.0,
                 physics_timestep=1 / 120.0, render_to_tensor=False, device_idx=0, use_pb_gui=False,):

        config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
        self.config = parse_config(config_data)
        if scene_id is not None:
            self.config["scene_id"] = scene_id

        if objects_find is not None:
            self.config["tar_objects"] = objects_find
            

        self.method = method
        self.config_filename = config_filename
        self.action_timestep = action_timestep
        self.physics_timestep = physics_timestep

        settings = MeshRendererSettings(enable_shadow=False, msaa=False, optimized=False)  # ,load_textures=True)
        self.simulator = Simulator(mode=mode, physics_timestep=physics_timestep, render_timestep=action_timestep,
                                   image_width=self.config.get("image_width", 128),
                                   image_height=self.config.get("image_height", 128),
                                   vertical_fov=self.config.get("vertical_fov", 90), device_idx=device_idx,
                                   rendering_settings=settings, use_pb_gui=use_pb_gui, )

        self.collision_ignore_body_b_ids = set(self.config.get("collision_ignore_body_b_ids", []))

        self.collision_ignore_link_a_ids = set(self.config.get("collision_ignore_link_a_ids", []))
        self.initial_pos_z_offset = self.config.get("initial_pos_z_offset", 0.1)
        self.initial_pos_z_offset_for_grasp = self.config.get("initial_pos_z_offset_for_grasp", 0.1)
        self.initial_pos_z_offset_2 = 0.05
        self.depth_high = self.config.get("depth_high", 5.0)
        self.depth_low = self.config.get("depth_low", 0.1)

        self.physically_simulated_openings = self.config.get("physically_simulated_openings", False)
        self.animate = self.config.get("animate", False)

        self.evaluate = self.config.get("evaluate", False)
        self.last_scene_id = self.config.get("scene_id", 'Rs_int')
        self.load_miscellaneous_variables()
        self.load(scene_id, objects_find)

        # Motion planner stuff
        self.mode = mode
        if self.physically_simulated_openings:
            self.motion_arm_planner = MotionPlanner(self, optimize_iter=10, full_observability_2d_planning=False,
                                                    collision_with_pb_2d_planning=False, visualize_2d_planning=False, visualize_2d_result=False,)

        

    def load_miscellaneous_variables(self):
        self.current_step = 0
        self.collision_step = 0
        self.current_episode = 0
        self.collision_links = []
        self.episode_counter = 0
        self.scene_reset_counter = 0
        self.aux_prob_counter = 0

        self.depth_high = self.config.get('depth_high', 2.5)
        self.depth_low = self.config.get('depth_low', 0.5)
        self.aux_task = self.config.get('use_aux_task', False)
        self.show_map = self.config.get('show_map', False)
        self.target_on_map = self.config.get('target_on_map', False)
        self.history_length = self.config.get('history_length', 1)
        self.substitute_polar = self.config.get('substitute_polar', False)
        self.aux_on_map = self.config.get('aux_on_map', False)
        self.set_polar_dist_zero = self.config.get('set_polar_dist_zero', False)
        self.reset_agent_pos = self.config.get('reset_agent_pos', False)
        self.multiple_envs = self.config.get('multiple_envs', False)
        self.resample_task = self.config.get('resample_task', False)
        self.increment_steps = self.config.get("increment_env_steps", False)

        self.interaction_failure_prob = 0.15
        self.drawer_pos_threshold = 0.13
        self.min_pixel_for_instances = 16
        self.min_episode_number = 25
        self.save_failed_eps = self.config.get("save_failed_eps", False)
        self.once_opening = self.config.get('once_opening', False)
        self.short_horizon_planning = self.config.get('short_horizon_planning', False)
        self.num_wps_cut = self.config.get("num_waypoints_cut", 10)
        self.normalize_hl_history = self.config.get("normalize_hl_history", False)

        self.numb_hl_actions = self.config.get("numb_hl_actions", 11.0)

        self.num_doors = self.config.get('num_doors', 4)
        self.num_cabinets = self.config.get('num_cabinets', 3)
        self.cam_fov = self.config.get('vertical_fov', 79.0)
        self.initial_camera_pitch = 0.0

        self.history_length_aux = self.config.get('history_length_aux', 10)
        self.aux_bin_number = 12

    def load(self, scene_id=None, num_tar_objects=None):
        config_data = yaml.load(open(self.config_filename, "r"), Loader=yaml.FullLoader)

        self.config = parse_config(config_data)
        if scene_id is not None:
            self.config["scene_id"] = scene_id
            if scene_id == "Benevolence_1_int":
                self.config["load_object_categories"] = ['shelf', 'door', 'sofa', 'table', 'window']

        if num_tar_objects is not None:
            self.config["tar_objects"] = num_tar_objects

        urdf_file = self.config.get("urdf_file", None)
        self.cracker_cabin_mixed = self.config.get('cracker_cabin_mixed', False)
        scene = InteractiveIndoorScene(
            self.config["scene_id"],
            urdf_file=urdf_file,
            waypoint_resolution=self.config.get("waypoint_resolution", 0.2),
            num_waypoints=self.config.get("num_waypoints", 10),
            build_graph=self.config.get("build_graph", False),
            trav_map_resolution=self.config.get("trav_map_resolution", 0.1),
            trav_map_erosion=self.config.get("trav_map_erosion", 2),
            trav_map_type=self.config.get("trav_map_type", "with_obj"),
            texture_randomization=False,
            object_randomization=False,
            object_randomization_idx=0,
            should_open_all_doors=self.config.get("should_open_all_doors", False),
            load_object_categories=self.config.get("load_object_categories", None),
            not_load_object_categories=self.config.get("not_load_object_categories", None),
            load_room_types=self.config.get("load_room_types", None),
            load_room_instances=self.config.get("load_room_instances", None),
            merge_fixed_links=self.config.get("merge_fixed_links", True)
            and not self.config.get("online_sampling", False), )

        self.simulator.import_scene(scene)
        robot_config = self.config["robot"]
        robot_name = robot_config.pop("name")
        if robot_name == "Fetch":
            robot = Fetch_DD(angular=self.config.get("angular_velocity"),
                             linear=self.config.get("linear_velocity"), **robot_config)
        else:
            robot = Locobot_DD(angular=self.config.get("angular_velocity"),
                               linear=self.config.get("linear_velocity"), **robot_config)
        # REGISTERED_ROBOTS[robot_name](**robot_config)
        self.simulator.import_object(robot)
        
        self.scene = self.simulator.scene
        self.robots = scene.robots
        
        self.mapping = MappingModule(self.config)

        self.task = HRLTask(self, self.config)
        self.task.load_cracker_objects(self)
        self.task.load_door_material(self)
        self.task.load_cabinet_objects(self)

        self.queue_for_task_resampling = []

        additional_information = 21+(16*12)
        tar_objects = 6
        observation_space = OrderedDict()
        coarse_map_size = 224
        fine_map_size = 84
        channels = 3
        self.add_frontier_exploration = self.config.get("add_frontier_exploration", False)
        self.add_exploration_policy = self.config.get("add_exploration_policy", False)
        self.add_invalid_action_masking = self.config.get("invalid_action_masking", False)
        observation_space["task_obs"] = self.build_obs_space(shape=(tar_objects+additional_information,), low=-np.inf, high=np.inf)
        observation_space['image'] = gym.spaces.box.Box(low=0, high=255, shape=(channels, fine_map_size, fine_map_size), dtype=np.uint8)
        observation_space['image_global'] = gym.spaces.box.Box(low=0, high=255, shape=(channels, coarse_map_size, coarse_map_size), dtype=np.uint8)

        obs_space_size = 0

        if self.add_frontier_exploration:
            obs_space_size += 1
        else:
            self.numb_hl_actions -= 1
            
        if self.add_exploration_policy:
            self.exploration_action = 0
            self.door_actions = [1,2,3,4]
            self.cabinet_actions = [5,6,7]
            self.cracker_actions = [8,9,10]
            if self.add_frontier_exploration:
                self.fr_action = 11
            else:
                self.fr_action = -1
            obs_space_size += 1
        else:
            self.numb_hl_actions -= 1
            self.exploration_action = -1
            self.door_actions = [0,1,2,3]
            self.cabinet_actions = [4,5,6]
            self.cracker_actions = [7,8,9]
            if self.add_frontier_exploration:
                self.fr_action = 10
            else:
                self.fr_action = -1

        observation_space['valid_actions'] = gym.spaces.box.Box(low=0, high=1.0, shape=(
                self.num_cabinets+self.num_doors+self.task.num_cracker+obs_space_size,), dtype=np.uint8)

        high_level_policy_obs_size = 28
        observation_space['task_obs_hl'] = gym.spaces.box.Box(
            low=-1.0, high=1.0, shape=(high_level_policy_obs_size,), dtype=np.uint8)  
        

        self.observation_space = gym.spaces.Dict(observation_space)

        self.load_action_space()

    def build_obs_space(self, shape, low, high):
        """
        Helper function that builds individual observation spaces
        """
        return gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)

    def load_action_space(self):
        """
        Load action space
        """
        self.action_space = self.robots[0].action_space_custom_xyz

    def reload_model(self, scene_id):
        """
        Reload another scene model
        This allows one to change the scene on the fly

        :param scene_id: new scene_id
        """
        self.config["scene_id"] = scene_id
        self.keep_doors_list = self.mapping.map_settings[scene_id]['doors']
        self.mapping.load_miscellaneous_map(scene_id)
        self.simulator.reload()
        self.load(scene_id)

    def filter_collision_links(self, collision_links):
        """
        Filter out collisions that should be ignored.

        :param collision_links: original collisions, a list of collisions
        :return: filtered collisions
        """
        # TODO: Improve this to accept multi-body robots.
        new_collision_links = []
        for item in collision_links:
            # ignore collision with body b
            if item[2] in self.collision_ignore_body_b_ids:
                continue

            # ignore collision with robot link a
            if item[3] in self.collision_ignore_link_a_ids:
                continue

            # ignore self collision with robot link a (body b is also robot itself)
            if item[2] == self.robots[0].base_link.body_id and item[4] in self.collision_ignore_link_a_ids:
                continue

            if item[2] in self.task.remove_collision_links:
                continue
            new_collision_links.append(item)
        return new_collision_links

    def run_simulation_planner(self, ignore_id):

        self.simulator.step()

        collision_links = [
            collision for bid in self.robots[0].get_body_ids() for collision in p.getContactPoints(bodyA=bid)
        ]
        new_collision_links = []

        for item in collision_links:
            # ignore collision with body b
            if item[2] in self.collision_ignore_body_b_ids:
                continue

            # ignore collision with robot link a
            if item[3] in self.collision_ignore_link_a_ids:
                continue

            # ignore self collision with robot link a (body b is also robot itself)
            if item[2] == self.robots[0].base_link.body_id and item[4] in self.collision_ignore_link_a_ids:
                continue

            new_collision_links.append(item)

        return new_collision_links

    def run_simulation(self):
        """
        Run simulation for one action timestep (same as one render timestep in Simulator class).

        :return: a list of collisions from the last physics timestep
        """
        self.simulator.step()
        collision_links = [
            collision for bid in self.robots[0].get_body_ids() for collision in p.getContactPoints(bodyA=bid)
        ]
        return self.filter_collision_links(collision_links)

    def get_depth(self, raw_depth):
        depth = -raw_depth[:, :, 2:3]
        # 0.0 is a special value for invalid entries
        depth[depth < self.depth_low] = 0.0
        depth[depth > self.depth_high] = 0.0
        
        return depth

    def get_sensor_obs(self):
        # with Profiler("Render"):
        if self.animate:
            frames = self.simulator.renderer.render_robot_cameras(modes=("seg", "3d","rgb"))
        else:
            frames = self.simulator.renderer.render_robot_cameras(modes=("seg", "3d"))

        
        depth = self.get_depth(frames[1])  
        
        seg = np.round(frames[0][:, :, 0:1] * MAX_CLASS_COUNT).astype(np.int32)
        
        if self.animate:
            return {"seg": seg, "depth": depth, "rgb": frames[2]}
        else:
            return {"seg": seg, "depth": depth}

    def real_cabinet_action(self, ind, action):
        
        collision, geo_dist, not_at_goal_pos = drive_to_cabinet(
            self, ind, ignore_id=[self.task.cabinets[ind].get_body_ids()[0]])
        self.reset_variables_ll()
        self.robots[0].keep_still()

        if not not_at_goal_pos:
            rob_pos = self.robots[0].base_link.get_position()[:2]
            # if agent "too close" to the cabinet already, just set its orientation looking towards the cabinet.
            current_yaw = self.robots[0].get_rpy()[2]
            tar_pos = self.task.target_pos_list[ind][:2]
            next_yaw_local = self.task.global_to_local(self, [tar_pos[0], tar_pos[1], 0.5])[:2]
            next_yaw_local_polar = np.array(cartesian_to_polar(next_yaw_local[0], next_yaw_local[1]))
            max_turn_angle = 0.1
            step_size = abs(int(np.round(next_yaw_local_polar[1] / max_turn_angle)))
            if step_size > 1 and self.animate:
                step_list = np.linspace(0.1,abs(next_yaw_local_polar[1]),num=step_size)
                for next_angle in step_list:
                    if next_yaw_local_polar[1] > 0.0:
                        set_base_values_with_z(self.robots[0].get_body_ids()[0], [rob_pos[0], rob_pos[1], current_yaw + next_angle],
                                    z=self.initial_pos_z_offset_2)
                    else:
                        set_base_values_with_z(self.robots[0].get_body_ids()[0], [rob_pos[0], rob_pos[1], current_yaw - next_angle],
                                    z=self.initial_pos_z_offset_2)
                    sensors = self.get_sensor_obs()
                    ego1, ego2 = self.mapping.run_mapping(self, sensors, action=None)
                    cv2.imwrite('data/vid{}/coarse/{}_{}'.format(self.current_episode,self.global_counter,'.png'),ego2.astype(np.uint8))
                    cv2.imwrite('data/vid{}/fine/{}_{}'.format(self.current_episode,self.global_counter,'.png'),ego1.astype(np.uint8))
                    cv2.imwrite('data/vid{}/rgb/{}_{}'.format(self.current_episode,self.global_counter,'.png'),cv2.cvtColor(sensors['rgb']*255,cv2.COLOR_RGB2BGR))
                    self.global_counter += 1
            target_yaw = current_yaw + next_yaw_local_polar[1]
            set_base_values_with_z(self.robots[0].get_body_ids()[0], [rob_pos[0],
                                                                      rob_pos[1], target_yaw], z=self.initial_pos_z_offset)

            

        if collision:
            self.robots[0].tuck()
            self.robots[0].reset()
            self.robots[0].keep_still()
        else:

            if True:

                cabinet_base_postion = self.task.target_pos_list[ind].copy()

                hit_normal = self.task.grasp_orientation[str(
                    self.task.target_pos_orient_list[ind])][0]  # [-0.00, 0.999, -0.04]
                pos_offset = self.task.grasp_orientation[str(self.task.target_pos_orient_list[ind])][1]

                ee_orientation = self.task.grasp_orientation[str(self.task.target_pos_orient_list[ind])][2]
                not_grasped = True
                trials = 0
                cabinet_base_postion = self.task.target_pos_list[ind].copy()
                
                while not_grasped and trials < 5:
                    trials += 1

                    # self.robots[0].reset()
                    # self.robots[0].keep_still()
                    self.robots[0].untuck()
                    
                    hit_pos = cabinet_base_postion.copy()
                    if pos_offset[0] != 0.0:
                        hit_pos[0] = hit_pos[0] + pos_offset[0] + np.random.uniform(low=-0.008, high=0.008)
                    else:
                        hit_pos[1] = hit_pos[1] + pos_offset[1] + np.random.uniform(low=-0.008, high=0.008)

                    hit_pos[2] = pos_offset[2] + np.random.uniform(low=-0.008, high=0.008)

                    # grasph_pose,tar_hand_pose = get_grasp_position_for_open(self.robots[0],self.task.cabinets[ind],should_open=True)#f"{self.task.cabinets[ind].name}_dof_rootd_Aa003_t")
                    plan = self.motion_arm_planner.plan_arm_push(hit_pos, -np.array(hit_normal), ee_orientation)

                    if plan is not None and len(plan) > 0:
                        
                        self.motion_arm_planner.execute_arm_push(plan, hit_pos, -np.array(hit_normal), ee_orientation)

                        self.robots[0].tuck()
                        self.robots[0].reset()
                        self.robots[0].keep_still()
                        

                        for _ in range(4):
                            self.simulator.step()
                        
                        j_state = self.task.cabinets[ind].get_joint_states()

                        # look at sum of all other drawers joint positions
                        sum_other_drawers = np.array([j_state[i][0]+j_state[i][1] for i in self.task.cabinets[ind].get_joint_states().keys() if i != f"{self.task.cabinets[ind].name}_dof_rootd_Aa003_t"]).sum()
                        
                        if sum_other_drawers >= self.drawer_pos_threshold:
                            close_all_drawers(self.task.cabinets[ind].get_body_ids()[0], [
                                              self.task.all_cabinet_joints[0]]+self.task.all_cabinet_joints[2::], self.task.ignore_plate_ids[ind])
                            p.changeDynamics(self.task.cabinets[action['hl_action']-self.cabinet_actions[0]].get_body_ids()
                                             [0], -1, activationState=p.ACTIVATION_STATE_WAKE_UP)
                           

                        if j_state[f"{self.task.cabinets[ind].name}_dof_rootd_Aa003_t"][0] > 0.09 and not sum_other_drawers >= self.drawer_pos_threshold:
                            not_grasped = False
                            pos_plate = self.task.plates[ind].get_base_link_position_orientation()[0]

                            if (l2_distance(pos_plate, self.task.target_pos_list[ind]) > 1.1 or (pos_plate[2] < 0.5 or pos_plate[2] > 0.6)) and int(ind) in self.task.cabinet_target_dict:
                                # try to respawn the plate into the cabinet
                                self.task.put_plate_in_drawer(self.task.plates[ind], self.task.cabinets[ind])

                            elif int(ind) in self.task.cabinet_target_dict:
                                
                                self.task.opened_cabinet.append(ind)
                                opened_in_this_step = True
                                
                                if int(ind) in self.task.cabinet_target_dict:
                                    self.rew_during_planner += 10.0  
                                    self.task.wanted_objects[self.task.cabinet_target_dict[ind]] = 0
                                    self.cabinet_indices.append(ind)
                            else:
                                tr_ps = self.task.target_pos_list[int(ind)]
                                tr_ps = self.mapping.world2map(tr_ps)
                                self.global_map[int(tr_ps[1])-4:int(tr_ps[1])+4, int(tr_ps[0]) -
                                    4:int(tr_ps[0])+4] = self.mapping.cabinet_marked
                                self.cabinet_indices.append(ind)
                                

                            

                    else:
                        pass

      

        return collision, geo_dist

    def cabinet_action(self, ind):

        collision, geo_dist, not_at_goal_pos = drive_to_cabinet(
            self, ind, ignore_id=[self.task.cabinets[ind].get_body_ids()[0]])
        self.reset_variables_ll()
        self.robots[0].keep_still()

        if not not_at_goal_pos:
            rob_pos = self.robots[0].base_link.get_position()[:2]
            # if agent "too close" to the cabinet already, just set its orientation looking towards the cabinet.
            current_yaw = self.robots[0].get_rpy()[2]
            tar_pos = self.task.target_pos_list[ind][:2]
            next_yaw_local = self.task.global_to_local(self, [tar_pos[0], tar_pos[1], 0.5])[:2]
            next_yaw_local_polar = np.array(cartesian_to_polar(next_yaw_local[0], next_yaw_local[1]))
            target_yaw = current_yaw + next_yaw_local_polar[1]
            set_base_values_with_z(self.robots[0].get_body_ids()[0], [rob_pos[0],
                                                                      rob_pos[1], target_yaw], z=self.initial_pos_z_offset)

            

        if collision:
            self.robots[0].tuck()
            self.robots[0].reset()
            self.robots[0].keep_still()


        else:

            # reject only the opening iteself.
            if np.random.uniform() < self.interaction_failure_prob or not_at_goal_pos:
                return collision, 2

            else:

                
                # in the case of wrong segmentation or just failing, segment object manually with ad-hoc solution
                if int(ind) in self.task.cabinet_target_dict:#self.task.wanted_objects[ind] == 1:
                    #the following target pos are essentially the cabinet spawn positions because plate pos are not needed here.
                    tr_ps = self.task.target_pos_list[int(ind)]
                    tr_ps = self.mapping.world2map(tr_ps)
                    self.global_map[int(tr_ps[1])-4:int(tr_ps[1])+4, int(tr_ps[0]) -
                                    4:int(tr_ps[0])+4] = self.mapping.cabinet_marked

                    if not (int(ind) in self.already_cabinet):
                        # take away the reward computation usually done in point_goal_REW.py since, there is no color coding but directly setting them as found
                        self.rew_during_planner += 10.0 
                        self.task.wanted_objects[self.task.cabinet_target_dict[ind]] = 0
                        self.already_cabinet.append(int(ind))
                else:
                    
                    tr_ps = self.task.target_pos_list[int(ind)]
                    tr_ps = self.mapping.world2map(tr_ps)
                    self.global_map[int(tr_ps[1])-4:int(tr_ps[1])+4, int(tr_ps[0]) -
                                    4:int(tr_ps[0])+4] = self.mapping.cabinet_marked
                
                
                self.cabinet_indices.append(ind)
                
        return collision, geo_dist

    def real_door_action(self, ind, door_name, real_door_index, selected_color, action):
        selected_color = self.mapping.original_door_colors[action['hl_action'] - self.door_actions[0]]
        real_door_index = np.argwhere(self.mapping.door_colors[:, 2] == selected_color[2])[0][0]
        door_name = self.task.door_index_to_key[real_door_index]
        door_id = self.task.door_dict[door_name][0]


        reject_door_opening = False

        # orientations mapped to directions doors open.
        door_opens_towards = {"-2.0": 0, "0.0": 0, "-0.0": 0, "2.0": 1, "-3.0": 1, "3.0": 1, "1.0": 0, "-1.0": 0}
        

        collision = True
        rob_pos = self.robots[0].base_link.get_position()[:2]
        door_pos = self.task.door_dict[door_name][1][:2]
        collision, geo_dist, not_at_goal_pos, which_side = drive_to_selected_door(real_door_index, self)

        if not_at_goal_pos or self.task.already_opened[door_name]:
            return collision, geo_dist, False

        # open door by pulling
        if which_side == door_opens_towards[str(self.task.door_dict[door_name][2])]:
            door_pos = self.task.door_dict[self.task.door_index_to_key[real_door_index]][1]

            hit_pos = self.task.offset_positions_door_knop[self.last_scene_id][door_name][0]
            if self.animate:
                rob_pos = self.robots[0].base_link.get_position()[:2]
                current_yaw = self.robots[0].get_rpy()[2]
                next_yaw_local = self.task.global_to_local(self, [hit_pos[0], hit_pos[1], 0.5])[:2]
                next_yaw_local_polar = np.array(cartesian_to_polar(next_yaw_local[0], next_yaw_local[1]))
                target_yaw = current_yaw + next_yaw_local_polar[1]

                max_turn_angle = 0.1
                step_size = abs(int(np.round(next_yaw_local_polar[1] / max_turn_angle)))
                if step_size > 1:
                    step_list = np.linspace(0.1,abs(next_yaw_local_polar[1]),num=step_size)
                    for next_angle in step_list:
                        if next_yaw_local_polar[1] > 0.0:
                            set_base_values_with_z(self.robots[0].get_body_ids()[0], [rob_pos[0], rob_pos[1], current_yaw + next_angle],
                                       z=self.initial_pos_z_offset_2)
                        else:
                            set_base_values_with_z(self.robots[0].get_body_ids()[0], [rob_pos[0], rob_pos[1], current_yaw - next_angle],
                                       z=self.initial_pos_z_offset_2)

                        sensors = self.get_sensor_obs()
                        ego1, ego2 = self.mapping.run_mapping(self, sensors, action=None)
                        #write images when animate is activated.
                        cv2.imwrite('data/vid{}/coarse/{}_{}'.format(self.current_episode,self.global_counter,'.png'),ego2.astype(np.uint8))
                        cv2.imwrite('data/vid{}/fine/{}_{}'.format(self.current_episode,self.global_counter,'.png'),ego1.astype(np.uint8))
                        cv2.imwrite('data/vid{}/rgb/{}_{}'.format(self.current_episode,self.global_counter,'.png'),cv2.cvtColor(sensors['rgb']*255,cv2.COLOR_RGB2BGR))
                        self.global_counter += 1
                        
                set_base_values_with_z(self.robots[0].get_body_ids()[0], [rob_pos[0],
                                                                      rob_pos[1], target_yaw], z=self.initial_pos_z_offset)
            else:
                rob_pos = self.robots[0].base_link.get_position()[:2]
                current_yaw = self.robots[0].get_rpy()[2]
                next_yaw_local = self.task.global_to_local(self, [hit_pos[0], hit_pos[1], 0.5])[:2]
                next_yaw_local_polar = np.array(cartesian_to_polar(next_yaw_local[0], next_yaw_local[1]))
                target_yaw = current_yaw + next_yaw_local_polar[1]
                set_base_values_with_z(self.robots[0].get_body_ids()[0], [rob_pos[0],
                                                                      rob_pos[1], target_yaw], z=self.initial_pos_z_offset)

            hit_normal = self.task.offset_positions_door_knop[self.last_scene_id][door_name][1]
            ee_orientation = self.task.offset_positions_door_knop[self.last_scene_id][door_name][2]
            trials = 0
            not_grasped = True
           
            if True:
                trials += 1

            
                self.robots[0].untuck()
                plan = self.motion_arm_planner.plan_arm_push(
                    hit_pos.copy(), -np.array(hit_normal), ee_orientation.copy())

                if plan is not None and len(plan) > 0:

                    self.motion_arm_planner.execute_arm_push(plan, hit_pos.copy(
                    ), -np.array(hit_normal), ee_orientation.copy(), doors=True, gripper_usage=True)

                self.robots[0].tuck()
                self.robots[0].reset()
                self.robots[0].keep_still()

                for _ in range(15):
                    #perform video smoothing as door is swining towards open state.
                    if self.animate:
                        sensors = self.get_sensor_obs()
                        ego1, ego2 = self.mapping.run_mapping(self, sensors, action=None)
                        cv2.imwrite('data/vid{}/coarse/{}_{}'.format(self.current_episode,self.global_counter,'.png'),ego2.astype(np.uint8))
                        cv2.imwrite('data/vid{}/fine/{}_{}'.format(self.current_episode,self.global_counter,'.png'),ego1.astype(np.uint8))
                        cv2.imwrite('data/vid{}/rgb/{}_{}'.format(self.current_episode,self.global_counter,'.png'),cv2.cvtColor(sensors['rgb']*255,cv2.COLOR_RGB2BGR))
                        self.global_counter += 1
                    self.simulator.step()

                door_joint_name = f"{door_name}_joint_2"

                joint_opening_state = self.scene.door_list[door_name][0].get_joint_states()[door_joint_name][0]
                
                thresh_ = 1.1
                if joint_opening_state < thresh_: 
                    
                    self.scene.restore_object_states_single_object(
                        self.scene.objects_by_name[door_name], self.scene.object_states[door_name])
                    p.changeDynamics(door_id, -1, activationState=p.ACTIVATION_STATE_WAKE_UP)
                    return collision, geo_dist, False
                else:
                    
                    if ind not in self.door_indices:
                        not_grasped = False
                        self.rew_during_planner += 3.0
                        self.door_indices.append(ind)
                      
                        self.task.already_opened[door_name] = True
                        selected_door_color_removal_indices = self.global_map[:, :, 2] == selected_color[2]
                        self.global_map[selected_door_color_removal_indices, :] = self.mapping.obstalce
                        return collision, geo_dist, True

        else:
            #open door by pushing
            door_pos = self.task.door_dict[self.task.door_index_to_key[real_door_index]][1]

            offset_pos_axis = self.task.pos_offset_axis_agent_to_doors[str(self.task.door_dict[door_name][2])]

            hit_pos = np.array(door_pos)
            if door_opens_towards[str(self.task.door_dict[door_name][2])] == 0:
                hit_pos[offset_pos_axis] += 0.1
            else:
                hit_pos[offset_pos_axis] -= 0.1


            if self.animate:
                rob_pos = self.robots[0].base_link.get_position()[:2]
                current_yaw = self.robots[0].get_rpy()[2]
                next_yaw_local = self.task.global_to_local(self, [hit_pos[0], hit_pos[1], 0.5])[:2]
                next_yaw_local_polar = np.array(cartesian_to_polar(next_yaw_local[0], next_yaw_local[1]))
                target_yaw = current_yaw + next_yaw_local_polar[1]

                max_turn_angle = 0.1
                step_size = abs(int(np.round(next_yaw_local_polar[1] / max_turn_angle)))
                if step_size > 1:
                    step_list = np.linspace(0.1,abs(next_yaw_local_polar[1]),num=step_size)
                    for next_angle in step_list:
                        if next_yaw_local_polar[1] > 0.0:
                            set_base_values_with_z(self.robots[0].get_body_ids()[0], [rob_pos[0], rob_pos[1], current_yaw + next_angle],
                                       z=self.initial_pos_z_offset_2)
                        else:
                            set_base_values_with_z(self.robots[0].get_body_ids()[0], [rob_pos[0], rob_pos[1], current_yaw - next_angle],
                                       z=self.initial_pos_z_offset_2)

                        sensors = self.get_sensor_obs()
                        ego1, ego2 = self.mapping.run_mapping(self, sensors, action=None)
                        cv2.imwrite('data/vid{}/coarse/{}_{}'.format(self.current_episode,self.global_counter,'.png'),ego2.astype(np.uint8))
                        cv2.imwrite('data/vid{}/fine/{}_{}'.format(self.current_episode,self.global_counter,'.png'),ego1.astype(np.uint8))
                        cv2.imwrite('data/vid{}/rgb/{}_{}'.format(self.current_episode,self.global_counter,'.png'),cv2.cvtColor(sensors['rgb']*255,cv2.COLOR_RGB2BGR))
                        self.global_counter += 1
                        
                set_base_values_with_z(self.robots[0].get_body_ids()[0], [rob_pos[0],
                                                                      rob_pos[1], target_yaw], z=self.initial_pos_z_offset)
            else:
                rob_pos = self.robots[0].base_link.get_position()[:2]
                current_yaw = self.robots[0].get_rpy()[2]
                next_yaw_local = self.task.global_to_local(self, [hit_pos[0], hit_pos[1], 0.5])[:2]
                next_yaw_local_polar = np.array(cartesian_to_polar(next_yaw_local[0], next_yaw_local[1]))
                target_yaw = current_yaw + next_yaw_local_polar[1]
                set_base_values_with_z(self.robots[0].get_body_ids()[0], [rob_pos[0],
                                                                      rob_pos[1], target_yaw], z=self.initial_pos_z_offset)

            

            hit_normal = self.task.offset_positions_door_knop[self.last_scene_id][door_name][1]
            # the orientation is arbitary here, therefore put None
            ee_orientation = None  

            trials = 0
            not_grasped = True
            
            if True:
                trials += 1
                
                self.robots[0].untuck()
                plan = self.motion_arm_planner.plan_arm_push(hit_pos.copy(), -np.array(hit_normal), ee_orientation)

                if plan is not None and len(plan) > 0:
                    self.motion_arm_planner.execute_arm_push(plan, hit_pos.copy(
                    ), -np.array(hit_normal), ee_orientation, doors=True, gripper_usage=False)

                self.robots[0].tuck()
                self.robots[0].reset()
                self.robots[0].keep_still()

                for _ in range(15):
                   
                    if self.animate:
                        sensors = self.get_sensor_obs()
                        ego1, ego2 = self.mapping.run_mapping(self, sensors, action=None)
                        cv2.imwrite('data/vid{}/coarse/{}_{}'.format(self.current_episode,self.global_counter,'.png'),ego2.astype(np.uint8))
                        cv2.imwrite('data/vid{}/fine/{}_{}'.format(self.current_episode,self.global_counter,'.png'),ego1.astype(np.uint8))
                        cv2.imwrite('data/vid{}/rgb/{}_{}'.format(self.current_episode,self.global_counter,'.png'),cv2.cvtColor(sensors['rgb']*255,cv2.COLOR_RGB2BGR))
                        self.global_counter += 1
                    self.simulator.step()

                door_joint_name = f"{door_name}_joint_2"

                joint_opening_state = self.scene.door_list[door_name][0].get_joint_states()[door_joint_name][0]
                

                thresh_ = 1.1
                if joint_opening_state < thresh_:  
                   
                    self.scene.restore_object_states_single_object(
                        self.scene.objects_by_name[door_name], self.scene.object_states[door_name])
                    p.changeDynamics(door_id, -1, activationState=p.ACTIVATION_STATE_WAKE_UP)
                    return collision, geo_dist, False
                else:
                   
                    if ind not in self.door_indices:
                        not_grasped = False
                        self.rew_during_planner += 3.0
                        self.door_indices.append(ind)
                        
                        self.task.already_opened[door_name] = True
                        selected_door_color_removal_indices = self.global_map[:, :, 2] == selected_color[2]
                        self.global_map[selected_door_color_removal_indices, :] = self.mapping.obstalce
                        return collision, geo_dist, True

        

    def door_action(self, ind, door_name, real_door_index, selected_color):
        
        self.reset_variables_ll()
        self.robots[0].keep_still()
        collision, geo_dist, not_at_goal_pos, which_side = drive_to_selected_door(real_door_index, self)

        door_id = self.task.door_dict[door_name][0]
        
        if collision:
            if not self.task.already_opened[door_name]:
                self.scene.door_list[door_name][0].main_body_is_fixed = True

                self.scene.restore_object_states_single_object(
                    self.scene.objects_by_name[door_name], self.scene.object_states[door_name])

                for joint_id in range(p.getNumJoints(door_id)):
                    p.changeDynamics(door_id, joint_id, mass=9999999.0, lateralFriction=0.1,
                                     spinningFriction=0.1, rollingFriction=0.1, frictionAnchor=True)
                p.changeDynamics(door_id, -1, activationState=p.ACTIVATION_STATE_WAKE_UP)

                self.robots[0].tuck()
                self.robots[0].reset()
                self.robots[0].keep_still()
                self.task.already_opened[door_name] = False

        else:

            if not not_at_goal_pos:
                rob_pos = self.robots[0].base_link.get_position()[:2]
                door_pos = self.task.door_dict[door_name][1][:2]
                current_yaw = self.robots[0].get_rpy()[2]
                next_yaw_local = self.task.global_to_local(self, [door_pos[0], door_pos[1], 0.5])[:2]
                next_yaw_local_polar = np.array(cartesian_to_polar(next_yaw_local[0], next_yaw_local[1]))
                target_yaw = current_yaw + next_yaw_local_polar[1]
                set_base_values_with_z(self.robots[0].get_body_ids()[0], [rob_pos[0],
                                                                          rob_pos[1], target_yaw], z=self.initial_pos_z_offset)

                

            # reject only the door opening iteself
            if np.random.uniform() < self.interaction_failure_prob or not_at_goal_pos:
                reject_door_opening = True
                
                return collision, 2
            else:
                _, open_iteration = self.scene.open_one_obj(door_id, mode="max")
                p.changeDynamics(door_id, -1, activationState=p.ACTIVATION_STATE_WAKE_UP)
                self.task.already_opened[door_name] = True

                
                if open_iteration > 2 and not self.task.already_opened[door_name]:
                    self.scene.restore_object_states_single_object(
                        self.scene.objects_by_name[door_name], self.scene.object_states[door_name])
                    for joint_id in range(p.getNumJoints(door_id)):
                        p.changeDynamics(door_id, joint_id, mass=9999999.0, lateralFriction=0.1,
                                         spinningFriction=0.1, rollingFriction=0.1, frictionAnchor=True)
                    p.changeDynamics(door_id, -1, activationState=p.ACTIVATION_STATE_WAKE_UP)

                    self.task.already_opened[door_name] = False
                else:
                    
                    if ind not in self.door_indices:
                        self.rew_during_planner += 3.0
                        self.door_indices.append(ind)
                        selected_door_color_removal_indices = self.global_map[:, :, 2] == selected_color[2]
                        self.global_map[selected_door_color_removal_indices, :] = self.mapping.obstalce
                       

        return collision, geo_dist

    def step(self, action):
        #rew_penalty serves as penalty reward when executing actions which are invalid according to the state while not using Invalid Action Masking.
        rew_penalty = False
        collision = 0.0
        #geo_dist describes the distance-traveled penalty received for the inividual subpolicy getting executed.
        geo_dist = 1.0
        self.rew_during_planner = 0.0
        
        if not self.add_invalid_action_masking and self.object_states[action['hl_action']] == 0:
            #wrong_command keeps track of how many times policy took invalid action.
            self.wrong_command += 1
            rew_penalty = True
            geo_dist = 2.0
        #SGoLam specific schemes
        elif action['hl_action'] == 12:
            geo_dist = 1
            rob_pos = self.robots[0].base_link.get_position()[:2]
            current_yaw = self.robots[0].get_rpy()[2]
            target_yaw = current_yaw + 0.075
            set_base_values_with_z(self.robots[0].get_body_ids()[0], [rob_pos[0],
                                                                        rob_pos[1], target_yaw], z=self.initial_pos_z_offset)
        #SGoLam specific schemes
        elif action['hl_action'] == 13:
            geo_dist = 1
            
            if self.collision_step_sgo == 1:
                #move forward
                self.robots[0].apply_action_old(np.array([1.0,0.0]))    
                self.collision_step_sgo = 0
            else:
                #turn agent
                self.collision_step_sgo += 1
                rob_pos = self.robots[0].base_link.get_position()[:2]
                current_yaw = self.robots[0].get_rpy()[2]
                target_yaw = current_yaw + 0.075
                set_base_values_with_z(self.robots[0].get_body_ids()[0], [rob_pos[0],
                                                                        rob_pos[1], target_yaw], z=self.initial_pos_z_offset)
        elif action['hl_action'] == self.exploration_action:
            
            self.robots[0].apply_action_old(action['action'])
            geo_dist = 0.5
            # exploration will be executed four time during training.
            if not self.increment_steps:
                self.current_step -= 0.75

        elif action['hl_action'] in self.cabinet_actions:
            
            ind = action['hl_action']-self.cabinet_actions[0]
            if self.physically_simulated_openings:
                collision, geo_dist = self.real_cabinet_action(ind, action)
            else:
                collision, geo_dist = self.cabinet_action(ind)

        elif action['hl_action'] in self.door_actions:
            # map the selected index to the actual door since the color varies from episode to episode
            ind = action['hl_action']-self.door_actions[0]
            selected_color = self.mapping.original_door_colors[ind]
            real_door_index = np.argwhere(self.mapping.door_colors[:, 2] == selected_color[2])[
                0][0]  
                
            door_name = self.task.door_index_to_key[real_door_index]

            if self.physically_simulated_openings:
                trials = 0
                succ = False
                while not succ and trials < 5:
                    collision, geo_dist, succ = self.real_door_action(
                        ind, door_name, real_door_index, selected_color, action)
                    trials += 1
            else:
                collision, geo_dist = self.door_action(ind, door_name, real_door_index, selected_color)

        elif action['hl_action'] in self.cracker_actions:
            ind = action['hl_action'] - self.cracker_actions[0]
            collision, geo_dist, not_at_goal_pos = drive_to_selected_cracker(self, ind)
            if collision:
                self.robots[0].tuck()
                self.robots[0].reset()
            else:
                self.cracker_indices.append(ind)
            self.reset_variables_ll()
            self.robots[0].keep_still()

        elif action['hl_action'] == self.fr_action: 
            if self.zero_masked_flag:
                self.robots[0].apply_action_old(np.random.uniform(-1,1,2))
                geo_dist = 0.5
                
            elif self.prev_frontier_point is not None:
                self.reset_variables_ll()
                self.robots[0].keep_still()
                goal_world = self.mapping.map2world(self.prev_frontier_point)
                collision, geo_dist, not_at_goal_pos = drive_to_frontier_point(self, [goal_world[0], goal_world[1]])

                if not not_at_goal_pos:
                    current_yaw = self.robots[0].get_rpy()[2]
                    rob_pos = self.robots[0].base_link.get_position()[:2]
                    tar_pos = goal_world
                    next_yaw_local = self.task.global_to_local(self, [tar_pos[0], tar_pos[1], 0.5])[:2]
                    next_yaw_local_polar = np.array(cartesian_to_polar(next_yaw_local[0], next_yaw_local[1]))
                    target_yaw = current_yaw + next_yaw_local_polar[1]
                    set_base_values_with_z(self.robots[0].get_body_ids()[0], [rob_pos[0], rob_pos[1], target_yaw],
                                           z=self.initial_pos_z_offset)

                if collision:
                    self.robots[0].tuck()
                    self.robots[0].reset()
                    self.robots[0].keep_still()

        else:
            print(f"Error action not known {action}")
            raise ValueError('UNKNOWN ACTION COMMAND !')
       
        collision_links = self.run_simulation()
        self.collision_links = collision_links
        c = int(len(collision_links) > 0)
        self.collision_step += c
        sensors = self.get_sensor_obs()

        if self.add_frontier_exploration:
            goal_position = explore(self.global_map, self.mapping.map_size[0], self.mapping.rob_pose)

            self.prev_frontier_point = goal_position

        ego1, ego2 = self.mapping.run_mapping(self, sensors, action)

        if self.show_map:

            cv2.imshow("fine-map", ego1.transpose(2, 0, 1).transpose(1, 2, 0).astype(np.uint8))
            cv2.waitKey(1)
            cv2.imshow("coarse-map", ego2.transpose(2, 0, 1).transpose(1, 2, 0).astype(np.uint8))
            cv2.waitKey(1)
            cv2.imshow("Static global map", self.global_map)
            cv2.waitKey(1)

            

        # check doors accoring to state observation
        doors_currently_on_map = np.where(np.array([(ego2[:, :, 2] == self.mapping.original_door_colors[i][2]).sum(
        ) for i in range(len(self.mapping.original_door_colors))]) > self.min_pixel_for_instances)  
        
        self.door_state = np.zeros(self.num_doors, dtype=int)
        self.door_state[doors_currently_on_map] = 1.0

        #self.door_state = self.door_state & self.task.succ_opened_doors

        #check cabinets according to state observation
        cabinets_currently_on_map = np.where(
            np.array([(ego2[:, :, 2] == self.mapping.cabinet_colors[i][2]).sum() for i in range(self.num_cabinets)]) > self.min_pixel_for_instances)

      
        self.cabinet_state = np.zeros(self.num_cabinets)
        self.cabinet_state[cabinets_currently_on_map] = 1.0

        cracker_currently_on_map = np.where(
            np.array([(ego2[:, :, 2] == self.mapping.cracker_colors[i][2]).sum() for i in range(self.task.num_cracker)]) > self.min_pixel_for_instances)
        self.cracker_state = np.zeros(self.task.num_cracker)
        self.cracker_state[cracker_currently_on_map] = 1.0

        if self.normalize_hl_history:
            self.last_hl_action_history.append(action['hl_action']/self.numb_hl_actions)
        else:
            self.last_hl_action_history.append(action['hl_action'])

        self.last_hl_action_history.pop(0)

        if self.once_opening:
            self.door_state[self.door_indices] = 0.0
            self.cabinet_state[self.cabinet_indices] = 0.0
            self.cracker_state[self.cracker_indices] = 0.0

        if self.add_frontier_exploration:
            if self.add_exploration_policy:
                self.object_states = np.concatenate(([1.0], self.door_state, self.cabinet_state, self.cracker_state, [
                                           1.0 if self.prev_frontier_point is not None else 0.0]))

                if action['hl_action'] == 0:
                    new_robot_pos = self.robots[0].get_position()[:2]
                    exploration_dist = l2_distance(self.robot_pos, new_robot_pos)
                    self.distance_taken.append(exploration_dist)
                    self.robot_pos = new_robot_pos
            else:
                self.object_states = np.concatenate((self.door_state, self.cabinet_state, self.cracker_state, [
                                           1.0 if self.prev_frontier_point is not None else 0.0]))
        else:
            if self.add_exploration_policy:
                self.object_states = np.concatenate(([1.0], self.door_state, self.cabinet_state, self.cracker_state))
                if action['hl_action'] == 0:
                    new_robot_pos = self.robots[0].get_position()[:2]
                    exploration_dist = l2_distance(self.robot_pos, new_robot_pos)
                    self.distance_taken.append(exploration_dist)
                    self.robot_pos = new_robot_pos
            else:
                self.object_states = np.concatenate((self.door_state, self.cabinet_state, self.cracker_state))



        info = {}
        info['penalty'] = rew_penalty
        
        if self.increment_steps:
            self.current_step += geo_dist
        else:
            self.current_step += 1

        info['discount_length'] = geo_dist  
        info['planner_collision'] = collision
        self.distance_planner.append(geo_dist)
        
        reward, info = self.task.get_reward(self, collision_links, action, info)
        self.task.step(self)
        reward += self.rew_during_planner


        done, info = self.task.get_termination(self, collision_links, action, info)

        self.populate_info(info, done)

        task_obs = self.task.get_task_obs(self)
        state = {}
        
        if self.object_states.sum() == 0:
            self.zero_masked_flag = True
            self.object_states[-1] = 1.0
        else:
            self.zero_masked_flag = False
        
        state['valid_actions'] = self.object_states

        state['image'] = ego1.transpose(2, 0, 1)

        state['image_global'] = ego2.transpose(2, 0, 1)

        state['task_obs_hl'] = np.concatenate(([task_obs[0], task_obs[1], np.array(self.coll_track).sum(), int(
            collision)], action['action'], self.task.wanted_objects[self.wanted_objects_mix], np.array(self.last_hl_action_history)))
        
        self.prev_locations.append(self.robots[0].base_link.get_position()[:2])
        self.coll_track.append(c)
        self.coll_track.pop(0)

        if len(self.prev_locations) >= self.history_length_aux:
            self.prev_locations.pop(0)
            self.aux_angle_track.pop(0)

        self.aux_angle_track.append(np.argmax(self.mapping.aux_action))

        var0 = np.var(np.array(self.prev_locations)[:, 0])
        var1 = np.var(np.array(self.prev_locations)[:, 1])

        state['task_obs'] = np.concatenate([self.mapping.aux_action, np.array(self.prev_aux_predictions).flatten(), [task_obs[0], task_obs[1], circvar(
            self.aux_angle_track), var0, var1], np.array([int(len(collision_links) > 0)]), np.array([np.array(self.coll_track).sum()]), action['action'], self.task.wanted_objects])

        self.prev_aux_predictions.append(self.mapping.aux_action)
        self.prev_aux_predictions.pop(0)

        if(self.episode_counter > self.min_episode_number and done and not info['success'] and self.resample_task):

            self.queue_for_task_resampling.append(
                (self.task.initial_pos, self.task.initial_orn, self.task.target_pos_list, self.task.initial_wanted, self.last_scene_id))

        if done and not info['success'] and self.save_failed_eps:
            img_cpy = self.global_map.copy()
            open_ind = np.argwhere(self.task.wanted_objects == 1)
            img_cpy[int(self.mapping.rob_pose[1])-4:int(self.mapping.rob_pose[1])+4,
                    int(self.mapping.rob_pose[0])-4:int(self.mapping.rob_pose[0])+4] = self.mapping.arrow

            for i in open_ind:

                tr_ps = self.task.target_pos_list[int(i)]
                tr_ps = self.mapping.world2map(tr_ps)
                img_cpy[int(tr_ps[1])-2:int(tr_ps[1])+2, int(tr_ps[0])-2:int(tr_ps[0])+2] = np.array([5, 128, 128])

        if self.animate:
            cv2.imwrite('data/vid{}/coarse/{}_{}'.format(self.current_episode,self.global_counter,'.png'),state['image_global'].transpose(1,2,0).astype(np.uint8))
            cv2.imwrite('data/vid{}/fine/{}_{}'.format(self.current_episode,self.global_counter,'.png'),state['image'].transpose(1,2,0).astype(np.uint8))
            cv2.imwrite('data/vid{}/rgb/{}_{}'.format(self.current_episode,self.global_counter,'.png'),cv2.cvtColor(sensors['rgb']*255,cv2.COLOR_RGB2BGR))
            self.global_counter += 1
        

        return state, reward, done, info

    def populate_info(self, info, done):
        """
        Populate info dictionary with any useful information
        """
        info["episode_length"] = self.current_step
        if done:
            info["collision_step"] = self.collision_step
            info["distance_planner"] = np.mean(np.array(self.distance_planner))
            info["distance_agent_general"] = np.array(self.distance_planner).sum() + np.array(self.distance_taken).sum()
            info['wrong_command'] = self.wrong_command

    def reset(self):

        # move robot away from the scene
        self.robots[0].set_position([100.0, 100.0, 100.0])
        # reset_agent assumes target positions to be already sampled
        while(not self.task.reset(self)):
            pass
        self.reset_variables_ll()

        # shuffle door colors
        self.mapping.reset(self)

        self.reset_variables()
        #random permuation for wanted_objects such that HL doesn't know how many plates are in some cabnets
        self.wanted_objects_mix = np.random.permutation(np.arange(6))

        self.task.object_distance = self.mapping.map_settings[self.last_scene_id]['object_dist']
        self.scene_reset_counter += 1
        self.aux_prob_counter += 1

        task_obs = self.task.get_task_obs(self)
        self.simulator.sync()

        sensors = self.get_sensor_obs()

        self.global_map = np.zeros((self.mapping.map_size[0], self.mapping.map_size[1], 3), dtype=np.uint8) * 255

        ego1, ego2 = self.mapping.run_mapping(self, sensors, None)

        self.prev_locations.append(self.robots[0].base_link.get_position()[:2])

        state = {}

        state['image'] = ego1.transpose(2, 0, 1)
        state['image_global'] = ego2.transpose(2, 0, 1)

        state['task_obs'] = np.concatenate([self.aux_pred_reset, np.array(self.prev_aux_predictions).flatten(), [
                                           task_obs[0], task_obs[1], 0.0, 0.0, 0.0], np.array([0.0]), np.array([0.0]), np.array([0.0, 0.0]), self.task.wanted_objects]) 

        self.robot_pos = self.robots[0].get_position()[:2]
        self.door_state = np.zeros(self.num_doors)
        self.cabinet_state = np.zeros(self.num_cabinets)
        self.cracker_state = np.zeros(self.task.num_cracker)

        if self.add_frontier_exploration:
            if self.add_exploration_policy:
                self.object_states = np.concatenate(
                    ([1.0], self.door_state, self.cabinet_state, self.cracker_state, [0.0]))
            else:
                self.object_states = np.concatenate(
                    (self.door_state, self.cabinet_state, self.cracker_state, [1.0]))
        else:
            if self.add_exploration_policy:
                self.object_states = np.concatenate(([1.0], self.door_state, self.cabinet_state, self.cracker_state))
            else:
                self.object_states = np.concatenate((self.door_state, self.cabinet_state, self.cracker_state))

        
        self.zero_masked_flag = False
        state['task_obs_hl'] = np.concatenate(
            ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], self.task.wanted_objects[self.wanted_objects_mix], np.array(self.last_hl_action_history)))

        state['valid_actions'] = self.object_states

        return state

    def reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode
        """
        self.global_counter = 0
        self.current_episode += 1
        self.episode_counter += 1
        self.current_step = 0
        self.rew_during_planner = 0.0
        self.collision_step = 0
        self.collision_links = []
        self.wrong_command = 0
        self.collision_step_sgo = 0
        self.already_cabinet = []
        self.distance_planner = []
        self.distance_taken = []
        self.prev_frontier_point = None
        self.last_hl_action_history = [-1.0]*16
        self.cabinet_indices = [] #[i[0] for i in np.argwhere(self.task.wanted_objects[:3] == 0)]
        self.cracker_indices = [i[0] for i in np.argwhere(self.task.wanted_objects[3::] == 0)]
        self.door_indices = []

    def reset_variables_ll(self):
        self.prev_locations = []
        self.coll_track = [0.0] * self.history_length_aux
        self.prev_aux_predictions = [np.zeros(self.aux_bin_number)]*self.history_length_aux
        self.aux_angle_track = []
        self.aux_pred_reset = np.zeros(self.aux_bin_number)+(1/self.aux_bin_number)

    def test_valid_position(self, obj, pos, orn=None, ignore_self_collision=False):
        """
        Test if the robot or the object can be placed with no collision.

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :param ignore_self_collision: whether the object's self-collisions should be ignored.
        :return: whether the position is valid
        """
        is_robot = isinstance(obj, BaseRobot)

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if is_robot:
            obj.reset()
            obj.keep_still()

        ignore_ids = obj.get_body_ids() if ignore_self_collision else []
        has_collision = any(self.check_collision(body_id, ignore_ids) for body_id in obj.get_body_ids())
        return not has_collision

    def set_pos_orn_with_z_offset(self, obj, pos, orn=None, offset=None):
        """
        Reset position and orientation for the robot or the object.

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :param offset: z offset
        """
        if orn is None:
            orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])

        if offset is None:
            offset = self.initial_pos_z_offset

        # first set the correct orientation
        obj.set_position_orientation(pos, quatToXYZW(euler2quat(*orn), "wxyz"))
        # get the AABB in this orientation
        lower, _ = obj.states[object_states.AABB].get_value()
        # Get the stable Z
        stable_z = pos[2] + (pos[2] - lower[2])
        # change the z-value of position with stable_z + additional offset
        # in case the surface is not perfect smooth (has bumps)
        obj.set_position([pos[0], pos[1], stable_z + offset])

    def check_collision(self, body_id, ignore_ids=[]):
        """
        Check whether the given body_id has collision after one simulator step

        :param body_id: pybullet body id
        :param ignore_ids: pybullet body ids to ignore collisions with
        :return: whether the given body_id has collision
        """
        self.simulator.step()
        collisions = [x for x in p.getContactPoints(bodyA=body_id) if x[2] not in ignore_ids]
        return len(collisions) > 0

    def land(self, obj, pos, orn):
        """
        Land the robot or the object onto the floor, given a valid position and orientation.

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        """
        is_robot = isinstance(obj, BaseRobot)

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if is_robot:
            obj.reset()
            obj.keep_still()

        land_success = False
        # land for maximum 1 second, should fall down ~5 meters
        max_simulator_step = int(1.0 / self.action_timestep)
        for _ in range(max_simulator_step):
            self.simulator.step()
            if any(len(p.getContactPoints(bodyA=body_id)) > 0 for body_id in obj.get_body_ids()):
                land_success = True
                break

        if is_robot:
            obj.reset()
            obj.keep_still()

    def run_sim_for_planner(self, ignore_id):
        collision_links = self.run_simulation_planner(ignore_id)
        if len(collision_links) > 0:
            return True

        sensors = self.get_sensor_obs()
        
        if self.animate:
            ego1, ego2 = self.mapping.run_mapping(self, sensors, action=None)
            cv2.imwrite('data/vid{}/coarse/{}_{}'.format(self.current_episode,self.global_counter,'.png'),ego2.astype(np.uint8))
            cv2.imwrite('data/vid{}/fine/{}_{}'.format(self.current_episode,self.global_counter,'.png'),ego1.astype(np.uint8))
            cv2.imwrite('data/vid{}/rgb/{}_{}'.format(self.current_episode,self.global_counter,'.png'),cv2.cvtColor(sensors['rgb']*255,cv2.COLOR_RGB2BGR))
            self.global_counter += 1
        else:
            self.mapping.run_mapping(self, sensors, action=None, no_ego_map=True)
        
        reward = self.task.reward_functions[0].get_reward(self.task, self, {})
        self.rew_during_planner += reward

        return False

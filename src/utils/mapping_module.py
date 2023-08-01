import numpy as np
import pybullet as p
from transforms3d.euler import euler2mat
import cv2
from scipy.special import softmax
from igibson.utils.utils import cartesian_to_polar
class MappingModule():

    def __init__(self, config):
        self.aux_bin_number = 12
        self.config = config
        self.physically_simulated_openings = self.config.get("physically_simulated_openings", False)
        self.load_miscellaneous_1()
        self.load_miscellaneous_2()
        self.load_miscellaneous_map(self.config.get("scene_id",'Rs_int'))

        
        self.offset_for_cut = 150
        self.aux_points = []
        step_size = 360 / self.aux_bin_number
        self.angle_to_bins = np.arange(step_size, 360 + step_size, step_size)  
        deg_steps = np.arange(0, 360, step_size)
        for i, deg in enumerate(deg_steps):
            ax = self.pol2cart(0.5, np.deg2rad(deg))
            ax = np.array([ax[0], ax[1], 0.0])
            self.aux_points.append(ax)

        self.door_positions = np.array([[],[],[],[]])
        self.check_radius = 6

    def world2map(self,xy):
        if (len(xy.shape) > 1):
            gpoints = np.array([self.offset, self.offset, self.offset]) + np.round(
                (xy + self.grid_offset) / self.grid_spacing)
            return gpoints

        else:
            x = (xy[0] + self.grid_offset[0]) / self.grid_spacing[0]
            y = (xy[1] + self.grid_offset[1]) / self.grid_spacing[1]

        return [np.round(x) + self.offset, np.round(y) + self.offset]

    def map2world(self,xy):
        return [(xy[1]*self.grid_spacing[0])-self.grid_offset[0],(xy[0]*self.grid_spacing[1])-self.grid_offset[1]]


    def pol2cart(self,rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)

    def run_mapping(self, env, sensors, action,no_ego_map=False,only_ego_map=False,stop_plate_recognition=False):

        #this is done in order to keep the color for found objects in articulated objects after the drawer or door
        #has been closed, otherwise it would be just over-painted by either ground or the object itself.
        pl1 = env.global_map[:, :, 0] == self.plate_colors[0][0]  
        pl2 = env.global_map[:, :, 0] == self.plate_colors[1][0]  
        pl3 = env.global_map[:, :, 0] == self.plate_colors[2][0]  
        pl4 = env.global_map[:, :, 0] == self.category_found[0]
        pl5 = env.global_map[:, :, 0] == self.cabinet_marked[0]

        d1 = env.global_map[:, :, 2] == self.door_colors[0][2]
        d2 = env.global_map[:, :, 2] == self.door_colors[1][2]
        d3 = env.global_map[:, :, 2] == self.door_colors[2][2]
        d4 = env.global_map[:, :, 2] == self.door_colors[3][2]


        sim_rob_position = env.robots[0].base_link.get_position()


        camera = env.robots[0]._links['eyes'].get_orientation()

        camera_translation = env.robots[0]._links['eyes'].get_position()


        camera_angles = p.getEulerFromQuaternion(camera)

        euler_mat = euler2mat(camera_angles[0], -camera_angles[1], camera_angles[2])

        self.rob_pose = self.world2map(np.array([sim_rob_position[0], sim_rob_position[1]]))

        if only_ego_map:
            return self.affine4map(env,env.robots[0].get_rpy()[2], camera_angles[2], action, euler_mat, camera_translation)

        point_cloud = self.pointcloud(sensors['depth'])

        w = (sensors['seg'] == 4).squeeze()

        point_cloud_walls = point_cloud[w, :]

        point_cloud_walls = euler_mat.dot(point_cloud_walls.T).T + camera_translation
        point_cloud_walls = self.world2map(point_cloud_walls).astype(np.uint16)

        f = (sensors['seg'] == 5).squeeze()

        point_cloud_floor = point_cloud[f, :]

        point_cloud_floor = euler_mat.dot(point_cloud_floor.T).T + camera_translation

        point_cloud_floor = self.world2map(point_cloud_floor).astype(np.uint16)

        path_indices = env.global_map[..., 2] == self.trace[2]
        
        try:
        
            env.global_map[point_cloud_floor[:, 1], point_cloud_floor[:, 0]] = self.obstalce

            env.global_map[point_cloud_walls[:, 1], point_cloud_walls[:, 0]] = self.floor

            mask = env.global_map[int(self.rob_pose[1]) - self.check_radius:int(self.rob_pose[1]) + self.check_radius, int(self.rob_pose[0]) \
            - self.check_radius:int(self.rob_pose[0]) + self.check_radius,0]==self.floor[0]
            
            env.global_map[int(self.rob_pose[1]) - self.check_radius:int(self.rob_pose[1]) + self.check_radius, int(self.rob_pose[0]) - \
            self.check_radius:int(self.rob_pose[0]) + self.check_radius,:][mask,:] = self.obstalce

            env.global_map[path_indices] = self.trace

            objects = (np.unique(sensors['seg'])).astype(int)

            self.draw_object_categories(env, objects, point_cloud, euler_mat, camera_translation, sensors['seg'],stop_plate_recognition=stop_plate_recognition)


            env.global_map[pl1, :] = self.plate_colors[0]
            env.global_map[pl2, :] = self.plate_colors[1]
            env.global_map[pl3, :] = self.plate_colors[2]
            env.global_map[pl4, :] = self.category_found 
            env.global_map[pl5, :] = self.cabinet_marked

            env.global_map[d1, :] = self.door_colors[0]
            env.global_map[d2, :] = self.door_colors[1]
            env.global_map[d3, :] = self.door_colors[2]
            env.global_map[d4, :] = self.door_colors[3]



            env.global_map[int(self.rob_pose[1]) - 2:int(self.rob_pose[1]) + 2,
            int(self.rob_pose[0]) - 2:int(self.rob_pose[0]) + 2] = self.trace

        except Exception as e:
            print("An exception occurred Mapping", e)
            print("CURRENT POSITION:",env.robots[0].base_link.get_position()[:2], "\n Orientation:",env.robots[0]._links['eyes'].get_orientation(),\
                "\n rest:",camera_translation, "scnene:",env.last_scene_id)
            
        
            
        

        if not no_ego_map:
            return self.affine4map(env,env.robots[0].get_rpy()[2], camera_angles[2], action, euler_mat, camera_translation)



    def reset(self,env):
        self.shuffle_indices = np.random.permutation(len(self.door_colors))

        self.door_colors = self.original_door_colors[self.shuffle_indices]
        self.grid_offset = self.map_settings[env.last_scene_id]['grid_offset']
        self.grid_spacing = self.map_settings[env.last_scene_id]['grid_spacing']
        self.offset = self.map_settings[env.last_scene_id]['offset']
        self.aux_action = env.aux_pred_reset
        self.load_miscellaneous_map(env.last_scene_id)


    def pad_img_to_fit_bbox(self, img, x1, x2, y1, y2):
        left = np.abs(np.minimum(0, y1))
        right = np.maximum(y2 - img.shape[0], 0)
        top = np.abs(np.minimum(0, x1))
        bottom = np.maximum(x2 - img.shape[1], 0)
        img = np.pad(img, ((left, right), (top, bottom), (0, 0)), mode="constant")

        y1 += left
        y2 += left
        x1 += top
        x2 += top
        return img, x1, x2, y1, y2


    def crop_fn(self, img: np.ndarray, center, output_size):
        h, w = np.array(output_size, dtype=int)
        x = int(center[0] - w / 2)
        y = int(center[1] - h / 2)

        y1, y2, x1, x2 = y, y + h, x, x + w
        if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
            img, x1, x2, y1, y2 = self.pad_img_to_fit_bbox(img, x1, x2, y1, y2)
        return img[y1:y2, x1:x2]

    
    def affine4map(self,env, rot_agent, rot_camera, action=None, euler_mat=None, camera_translation=None):

        img_copy = env.global_map.copy()
        if env.prev_frontier_point is not None and env.add_frontier_exploration:
            
            tar_pos_world = self.map2world(env.prev_frontier_point)
            bla = env.task.global_to_local(env, [tar_pos_world[0], tar_pos_world[1], 0.5])[:2]
            
            polar_local = np.array(cartesian_to_polar(bla[0],bla[1]))
            #rough approx., as the max radius of displayed map is at 7.2meter
            if polar_local[0] > 6.8:
                
                polar_local[0] = min(polar_local[0],6.8)
                cart_local_mod = self.pol2cart(polar_local[0],polar_local[1])
            
                cart_global = env.task.local_to_global(env,[cart_local_mod[0],cart_local_mod[1],0.5])
            
                cart_map = self.world2map(cart_global)
                img_copy[int(cart_map[1])-7:int(cart_map[1])+7,int(cart_map[0])-7:int(cart_map[0])+7] = self.frontier_point_pruned

            else:
                img_copy[env.prev_frontier_point[0]-7:env.prev_frontier_point[0]+7,env.prev_frontier_point[1]-7:env.prev_frontier_point[1]+7] = self.frontier_point
                
        
        if action is not None:

            if action['hl_action'] == env.exploration_action:
                self.aux_action = softmax(action['aux_angle'])

            p_ax = self.world2map(euler_mat.dot(np.array(self.aux_points).T).T + camera_translation)
            for i, p in enumerate(p_ax):
                img_copy[int(p[1]) - 2:int(p[1]) + 2, int(p[0]) - 2:int(p[0]) + 2, 1::] = self.aux_pred[1::]
                img_copy[int(p[1]) - 2:int(p[1]) + 2, int(p[0]) - 2:int(p[0]) + 2, 0] = self.aux_pred[0] * \
                                                                                            self.aux_action[i]

        pos = self.rob_pose

        cropped_map = self.crop_fn(img_copy, center=pos, output_size=(
            self.cut_out_size2[0] + self.offset_for_cut, self.cut_out_size2[1] + self.offset_for_cut))

        w, h, _ = cropped_map.shape
        center = (h / 2, w / 2)
        M = cv2.getRotationMatrix2D(center, np.rad2deg(rot_agent) + 90.0, 1.0)
        ego_map = cv2.warpAffine(cropped_map, M, (h, w),
                             flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0))  

        # when putting the circle here, the global map has a sharper version in which the circle does not appear as rectangle in some cases
        cv2.circle(
            img=ego_map,
            center=((self.cut_out_size2[0] + self.offset_for_cut) // 2, (self.cut_out_size2[1] + self.offset_for_cut) // 2),
            radius=int(6),
            color=(int(self.arrow[0]), int(self.arrow[1]), int(self.arrow[2])),
            thickness=-1,
        )

        ego_map_local = self.crop_fn(ego_map, center=(ego_map.shape[0] / 2, ego_map.shape[1] / 2),
                                 output_size=(self.cut_out_size[0], self.cut_out_size[1]))
        ego_map = self.crop_fn(ego_map, center=(ego_map.shape[0] / 2, ego_map.shape[1] / 2),
                           output_size=(self.cut_out_size2[0], self.cut_out_size2[1]))

        ego_map_global = cv2.resize(ego_map, (self.downsample_size, self.downsample_size), interpolation=cv2.INTER_NEAREST)

        return ego_map_local, ego_map_global


    def pointcloud(self, depth):
        depth = depth.squeeze()
        rows, cols = depth.shape

        px, py = (rows / 2, cols / 2)
        hfov = self.cam_fov / 360. * 2. * np.pi
        fx = rows / (2. * np.tan(hfov / 2.))

        vfov = 2. * np.arctan(np.tan(hfov / 2) * cols / cols)
        fy = cols / (2. * np.tan(vfov / 2.))

        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        valid = (depth > 0)  # & (depth < 255)
        z = np.where(valid, depth, 0.0)
        x = np.where(valid, z * (c - (rows / 2)) / fx, 0)
        y = np.where(valid, z * (r - (cols / 2)) / fy, 0)
        return np.dstack((z, -x, y))


    def draw_object_categories(self, env, objects_current_frame, point_cloud, euler_mat, camera_translation, seg, stop_plate_recognition=False):

        for ob in objects_current_frame:
            # 0 is outer space, 2- robot 
            if (ob in [0,1,2,3,4,5]):
                continue

            point_cloud_category = point_cloud[(seg == ob).squeeze(), :]
            
            point_cloud_category = euler_mat.dot(point_cloud_category.T).T + camera_translation
            point_cloud_category = self.world2map(point_cloud_category).astype(np.uint16)
           
            if (ob == 320):
                env.global_map[point_cloud_category[:, 1], point_cloud_category[:, 0]] = self.category_picture
            elif ob in env.task.cracker_ids:
                ind = env.task.cats_to_ind[ob]

                env.global_map[point_cloud_category[:, 1], point_cloud_category[:, 0]] = self.cracker_colors[ind-3]

                if env.task.wanted_objects[ind] == 0.0 and point_cloud_category.sum() > 19:
                    tr_ps = env.task.target_pos_list[int(ind)]
                    tr_ps = self.world2map(tr_ps)
                    env.global_map[int(tr_ps[1])-5:int(tr_ps[1])+5,int(tr_ps[0])-3:int(tr_ps[0])+2] = self.category_found
                    
            elif ob in env.task.forbidden_door_sem_ids:
                env.global_map[point_cloud_category[:, 1], point_cloud_category[:, 0]] = self.forbidden_door_color
            elif (ob in env.task.door_sem_ids):
                ind = env.task.door_cat_to_ind[ob]
                env.global_map[point_cloud_category[:, 1], point_cloud_category[:, 0]] = self.door_colors[
                    ind]  
                
        
                key = env.task.door_index_to_key[ind]
                offset_pos_axis = env.task.pos_offset_axis_agent_to_doors[str(env.task.door_dict[key][2])]
                pos = env.task.door_dict[key][1]
                pos = self.world2map(np.array([pos[0], pos[1]]))
                if not env.task.already_opened[env.task.door_index_to_key[ind]] and point_cloud_category.sum() > 25:
                    if offset_pos_axis == 1:
                        env.global_map[int(pos[1]) - 2:int(pos[1]) + 2,int(pos[0]) - 8:int(pos[0]) + 8] = self.door_colors[ind]
                    else:
                        env.global_map[int(pos[1]) - 8:int(pos[1]) + 8,int(pos[0]) - 2:int(pos[0]) + 2] = self.door_colors[ind]
                #this parts enlarges the door in the semantic map too make it thicker (can act counterintuitvely for the Exploration Policy)
                else:
                    if offset_pos_axis == 0:
                        pos = np.array(env.task.door_dict[key][1])
                        pos[1] -= 0.42
                        pos = self.world2map(np.array([pos[0], pos[1]]))
                        env.global_map[int(pos[1]) - 2:int(pos[1]) + 2,int(pos[0]) - 5:int(pos[0]) + 5] = self.door_colors[ind]
                    else:
                        pos = np.array(env.task.door_dict[key][1])
                        pos[0] -= 0.42
                        pos = self.world2map(np.array([pos[0], pos[1]]))
                        env.global_map[int(pos[1]) - 5:int(pos[1]) + 5,int(pos[0]) - 2:int(pos[0]) + 2] = self.door_colors[ind]
                
                
                
            elif (ob in env.task.cabinet_ids):
                ind = env.task.cabinet_cat_to_ind[ob]
                
                env.global_map[point_cloud_category[:, 1], point_cloud_category[:, 0]] = self.cabinet_colors[ind]
                
                    
            elif (ob in env.task.plate_ids and not stop_plate_recognition):
                ind = env.task.plate_cat_to_ind[ob]
               
                tr_ps = env.task.target_pos_list[int(ind)]
                tr_ps = self.world2map(tr_ps)

                
                
                if point_cloud_category.sum() > 1024:
                    env.global_map[int(tr_ps[1])-4:int(tr_ps[1])+4,int(tr_ps[0])-4:int(tr_ps[0])+4] = self.cabinet_marked

               
            elif (ob == 323):
                env.global_map[point_cloud_category[:, 1], point_cloud_category[:, 0]] = self.category_console_table
            elif (ob in [317 ,392]):
                env.global_map[point_cloud_category[:, 1], point_cloud_category[:, 0]] = self.arbitary_category
            



    def load_miscellaneous_2(self):
        # Images have BGR format
        self.floor = np.array([0, 255, 0])
        self.obstalce = np.array([255, 0, 0])
        self.trace = np.array([164, 0, 255])
        self.arrow = np.array([0, 128, 255])

        self.scene_reset_counter = 0
        self.aux_prob_counter = 0
        self.arbitary_category = np.array([95, 190, 45])
        self.cabinet_colors = [np.array([111, 32, 177]), np.array([26, 144, 214]),np.array([177, 155, 112])]

        self.categorie_array = [np.array([64, 64, 64]), np.array([32, 152, 196]), np.array([12, 48, 96]),
                                np.array([102, 32, 77]), np.array([126, 55, 133]), np.array([140, 109, 84]), np.array([112, 101, 90]), np.array([155, 112, 101])]

        self.plate_colors = [np.array([64, 64, 64]), np.array([32, 152, 196]), np.array([12, 48, 96])]

        self.cracker_colors = [np.array([102, 32, 77]), np.array([126, 55, 133]), np.array([140, 109, 84])]
        

        self.door_colors = np.array([np.array([81,255,222]), np.array([91, 255, 144]), np.array([117, 255, 106]), np.array([199, 255, 27])])
        self.forbidden_door_color = np.array([77, 12, 128])
        self.num_door_color_indices = np.arange(self.config.get('num_door_colors', 4))
  
        
        self.original_door_colors = self.door_colors.copy()

        self.category_found = np.array([249, 192, 203])

        self.cabinet_marked = np.array([200, 200, 10])
        self.cabinet_marked_empty = np.array([244, 44, 5])

        self.category_picture = np.array([13, 66, 220])

        self.category_console_table = np.array([95, 190, 45])
        self.aux_pred = np.array([192, 25, 79])

        self.frontier_point = np.array([159, 79, 122])
        self.frontier_point_pruned = np.array([159, 161, 122])
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

        self.downsample_size = self.config.get('global_map_size', 128)
        self.resample_task = self.config.get('resample_task', True)
        

        self.cam_fov = self.config.get('vertical_fov', 79.0)

        self.initial_camera_pitch = 0.0


    def load_miscellaneous_map(self, scene_id):

        map_settings_size = self.map_settings[scene_id]['map_size']
        self.map_size = (map_settings_size, map_settings_size)  # 164 neu 142

        self.map_size_2 = (128, 128)


        self.cut_out_size = (84, 84)
        self.x_off1 = self.map_size[0] - self.cut_out_size[0]
        self.x_off2 = int(self.map_size[0] - (self.x_off1 // 2))
        self.x_off1 = self.x_off1 // 2

        self.y_off1 = self.map_size[1] - self.cut_out_size[1]
        self.y_off2 = int(self.map_size[1] - (self.y_off1 // 2))
        self.y_off1 = self.y_off1 // 2

        # greater map
        self.cut_out_size2 = (420, 420)
        self.x_off11 = self.map_size[0] - self.cut_out_size2[0]
        self.x_off22 = int(self.map_size[0] - (self.x_off11 // 2))
        self.x_off11 = self.x_off11 // 2

        self.y_off11 = self.map_size[1] - self.cut_out_size2[1]
        self.y_off22 = int(self.map_size[1] - (self.y_off11 // 2))
        self.y_off11 = self.y_off11 // 2

    def load_miscellaneous_1(self):

        self.grid_res = 0.033  
        self.map_settings = {}
      
        #Test set
        #-----------------------------------------------------------------------------------------------------
        #-----------------------------------------------------------------------------------------------------
        #-----------------------------------------------------------------------------------------------------
        self.map_settings['Benevolence_1_int'] = {'grid_offset': np.array([5.5, 10.5, 15.1]),
                                                  'grid_spacing': np.array([self.grid_res, self.grid_res, 0.1]),
                                                  'offset': 0, 'object_dist': 1.4, 'door_dist': 1.4,'object_dist_cabinets':1.4,
                                                  'doors':['door_52','door_54','door_55'],'forbidden_doors': ['door_52'], "map_size": 450,
                                                  "cab_spawns":{"0":[(0.0,0.0,3.1),(0,0.97,-7.78,1.3)],\
                                                  "1":[(0.0, 0.0, 0.0),(0,-3.35,-6.9,-5.8)],
                                                  "2":[(0.0, 0.0, -1.55),(1,-1.55,0.99,1.4)]}}
        self.map_settings['Benevolence_2_int'] = {'grid_offset': np.array([5.5, 10.5, 15.1]),
                                                  'grid_spacing': np.array([self.grid_res, self.grid_res, 0.1]),
                                                  'offset': 0, 'object_dist': 1.7, 'door_dist': 1.4,'object_dist_cabinets':1.2,
                                                  'doors':["door_35","door_41","door_43"],'forbidden_doors':[],"map_size": 450,
                                                  "cab_spawns":{"0":[(0.0,0.0,0.0),(0,-3.41,-1.16,1.42)],
                                                  "1":[(0.0, 0.0, 3.1),(0,1.13,-8.0,1.45)],
                                                  "2":[(0.0, 0.0, 3.1),(0,1.13,-8.0,1.45)]}} 
        
        self.map_settings['Pomaria_1_int'] = {'grid_offset': np.array([15.0, 7.0, 15.1]),
                                              'grid_spacing': np.array([self.grid_res, self.grid_res, 0.1]),
                                              'offset': 0, 'object_dist': 2.1, 'door_dist': 1.4,'object_dist_cabinets':1.3,
                                              'doors':['door_65','door_70','door_72','door_73'],'forbidden_doors':['door_65','door_70'], "map_size": 550,
                                              "cab_spawns":{"0":[(0.0,0.0,3.1),(0,0.68,-3.4,3.85)],
                                                  "1":[(0.0, 0.0, -1.55),(1,-11.98,-7.8,2.65),(1,-6.98,0.7,3.8)],
                                                  "2":[(0.0, 0.0, 1.55),(1,-11.84,-7.14,-2.6)]}}

        self.map_settings['Pomaria_2_int'] = {'grid_offset': np.array([7.5, 7.5, 15.1]),
                                              'grid_spacing': np.array([self.grid_res, self.grid_res, 0.1]),
                                              'offset': 0, 'object_dist': 1.24, 'door_dist': 1.4, 'object_dist_cabinets':1.2,
                                              'doors':["door_29","door_32"],'forbidden_doors':[],
                                              "map_size": 450,
                                              "cab_spawns":{"0":[(0.0,0.0,3.1),(0,0.68,-2.71,0.35)],
                                              "1":[(0.0, 0.0, 1.55),(1,-4.85,0.68,-2.71)],
                                              "2":[(0.0, 0.0, 1.55),(1,-0.74,0.16,1.9)],
                                              "3":[(0.0,0.0,0.0),(0,-4.85,-2.71,0.65)]}}
        
        self.map_settings['Merom_1_int'] = {'grid_offset': np.array([10.0, 7.0, 15.1]),
                                            'grid_spacing': np.array([self.grid_res, self.grid_res, 0.1]), 'offset': 0,
                                            'object_dist': 1.6, 'door_dist': 1.4,'object_dist_cabinets':1.2,
                                            'doors':['door_74','door_93','door_85','door_86','door_87','door_88'],'forbidden_doors':['door_74','door_93'],
                                            "map_size": 650,
                                            "cab_spawns":{"0":[(0.0,0.0,0.0),(0,-2.2,-1.41,8.32)],
                                              "1":[(0.0, 0.0, 1.55),(1,-2.1,4.69,-1.48)],
                                              "2":[(0.0,0.0,3.1),(0,4.7,-1.48,7.47)]}}

        self.map_settings['Wainscott_0_int'] = {'grid_offset': np.array([8.5, 8.0, 15.1]),
                                                'grid_spacing': np.array([self.grid_res, self.grid_res, 0.1]),
                                                'offset': 0, 'object_dist': 1.6,  'object_dist': 2.0,'object_dist_cabinets':1.2,
                                                'doors':['door_126','door_128','door_132','door_135','door_134','door_136','door_137'],'forbidden_doors':['door_126','door_128','door_132','door_135'],
                                                "map_size": 750,
                                                "cab_spawns":{"0":[(0.0,0.0,0.0),(0,-4.34,-4.8,12.6)],
                                                "1":[(0.0,0.0,3.1),(0,0.85,-0.1,5.04),(0,6.5,5.02,12.51)]}}  # <--- massive map

        self.map_settings['Beechwood_0_int'] = {'grid_offset': np.array([13.5, 8.5, 15.1]),
                                                'grid_spacing': np.array([self.grid_res, self.grid_res, 0.1]),
                                                'offset': 0, 'object_dist': 2.5, 'door_dist': 2.5,"object_dist_cabinets":2.0,
                                                
                                                'doors':['door_93','door_109','door_97','door_98','door_101','door_102'],'forbidden_doors':['door_93','door_109'], "map_size": 550,
                                                "cab_spawns":{"0":[(0.0,0.0,0.0),(0,-10.0,-2.8,5.04)],
                                                "1":[(0.0, 0.0, 1.55),(1,-10.7,-7.72,-2.75),(1,-8.14,-4.65,1.4),(1,-6.56,1.3,-5.6)],
                                                "2":[(0.0,0.0,3.1),(0,-7.7,-2.6,0.51),(0,1.3,-5.6,2.76)],
                                                "3":[(0.0,0.0,-1.55),(1,-7.0,-4.41,-1.855)]}}  # 2.5
       

        

        #Training set
        #-----------------------------------------------------------------------------------------------------
        #-----------------------------------------------------------------------------------------------------
        #-----------------------------------------------------------------------------------------------------

        self.map_settings['Merom_0_int'] = {'grid_offset': np.array([3.3, 2.2, 15.1]),
                                            'grid_spacing': np.array([self.grid_res, self.grid_res, 0.1]), 'offset': 0,
                                            'object_dist': 1.6,'object_dist_cabinets':1.35, 'door_dist': 1.4, 'doors': ['door_63','door_64','door_67','door_60']\
                                            ,'forbidden_doors':['door_60'], "map_size": 400,"cab_spawns":{
                                            "0":[(0.0,0.0,0.0),(0,-2.3,-1.1,6.5),(0,1.5,-1.1,2.8)],\
                                            "1":[(0.0, 0.0, 3.1),(0,0.75,-1.3,1.0),(0,0.75,3.0,3.4),(0,4.7,1.0,2.8),(0,4.7,6.7,8.0)],
                                            "2":[(0.0, 0.0, 1.55),(1,-1.4,0.5,-1.55)]}}#,(0,4.7,1.0,2.8) - led to collisions

        self.map_settings['Benevolence_0_int'] = {'grid_offset': np.array([5.5, 9.5, 15.1]),
                                                  'grid_spacing': np.array([self.grid_res, self.grid_res, 0.1]),
                                                  'offset': 0, 'object_dist': 0.2,'object_dist_cabinets':0.7, 'door_dist': 1.4,
                                                  'doors': ['door_9', 'door_12', 'door_13','door_11'],'forbidden_doors':['door_9', 'door_12', 'door_13'], "map_size": 450,\
                                                  "cab_spawns":{
                                                  "0":[(0.0,0.0,0.0),(0,-3.3,-5.2,-7.2)],\
                                                  "1":[(0.0, 0.0, 1.55),(1,-3.4,-2.5,-7.1)],
                                                  "2":[(0.0, 0.0, 3.1),(-1,-0.612, -5.69),(-1,0.95, -6.275)],\
                                                 "3":[(0.0, 0.0, 1.55),(-1,-3.2,-4.23)]}}

        self.map_settings['Pomaria_0_int'] = {'grid_offset': np.array([15.0, 6.5, 15.1]),
                                              'grid_spacing': np.array([self.grid_res, self.grid_res, 0.1]),
                                              'offset': 0, 'object_dist': 1.6,'object_dist_cabinets':1.35, 'door_dist': 1.4,
                                              'doors': ['door_41', 'door_42','door_44','door_46','door_40'],'forbidden_doors': \
                                              ['door_41', 'door_42'], "map_size": 550,\
                                              "cab_spawns":{\
                                            "0":[(0.0,0.0,0.0),(0,-1.63,-2.55,0.5),(0,-4.5,-2.5,-0.85),(0,-4.68,2.97,3.7)],\
                                            "1":[(0.0, 0.0, 1.55),(1,-1.68,0.8,-2.5)],\
                                            "2":[(0.0, 0.0, 0.0),(-1,-8.14,-1.435)]}}#

        self.map_settings['Wainscott_1_int'] = {'grid_offset': np.array([8.0, 8.0, 15.1]),
                                                'grid_spacing': np.array([self.grid_res, self.grid_res, 0.1]),
                                                'offset': 0, 'object_dist': 2.5,'object_dist_cabinets':2.0, 'door_dist': 1.4, 'doors': ['door_86','door_89','door_88','door_95'],'forbidden_doors': [],
                                                "map_size": 700,\
                                                "cab_spawns":{\
                                            "0":[(0.0,0.0,0.0),(0,-4.37,-6.34,-0.1),(0,-4.3,4.78,6.93),(0,-4.18,7.9,9.1)],\
                                            "1":[(0.0, 0.0, -1.55),(1,2.27,4.82,12.2),(1,3.9,6.53,7.7)],\
                                            "2":[(0.0,0.0,3.1),(0,7.0,-0.2,2.1)]}} 
        
        

        self.map_settings['Rs_int'] = {'grid_offset': np.array([6.0, 5.5, 15.1]),
                                       'grid_spacing': np.array([self.grid_res, self.grid_res, 0.1]), 'offset': 0,
                                       'object_dist': 1.0,'object_dist_cabinets':1.4, 'doors': ['door_54','door_52'],'forbidden_doors': ['door_54'], "door_dist": 1.0, "map_size": 450,\
                                       "cab_spawns":{"0":[(0.0,0.0,3.1),(0,1.59,-2.99,0.713),(0,-2.41,-1.59,0.14)],\
                                            "1":[(0.0, 0.0, 0.0),(0,-3.72,-1.64,1.06)],\
                                            "2":[(0.0, 0.0, 0.0),(-1,-1.725 , 2.433)],\
                                            "3":[(0.0, 0.0, 1.55),(-1,0.131 ,-3.778)]}}
                                            

        self.map_settings['Ihlen_0_int'] = {'grid_offset': np.array([5.5, 3.0, 15.1]),
                                            'grid_spacing': np.array([self.grid_res, self.grid_res, 0.1]), 'offset': 0,
                                            'object_dist': 2.5,'object_dist_cabinets':2.5, 'door_dist': 1.4, 'doors': ['door_42','door_46','door_47'],'forbidden_doors': ['door_42'], "map_size": 450,
                                            "cab_spawns":{
                                            "0":[(0.0,0.0,3.1),(0,4.5,-0.83,4.1),(0,4.5,6.0,7.1),(0,0.9,-0.5,4.2)],\
                                            "1":[(0.0, 0.0, 0.0),(0,1.6,-1.1,3.4),(0,-4.6,7.8,10.4),(0,-4.6,0.84,4.07)],\
                                            "2":[(0.0, 0.0, 1.55),(1,1.8 ,4.46,-0.95),(1,-4.6 ,-1.9,0.85),(1,-1.07 ,0.5,-1.17)],\
                                            "3":[(0.0, 0.0, -1.55),(1,-4.43 ,1.73,10.4),(1,-4.5 ,-2.24,4.43)]}}

        self.map_settings['Beechwood_1_int'] = {'grid_offset': np.array([11.5, 8.5, 15.1]),
                                                'grid_spacing': np.array([self.grid_res, self.grid_res, 0.1]),
                                                'offset': 0, 'object_dist': 2.5,'object_dist_cabinets':2.5, 'door_dist': 3.1, 'doors': ['door_80','door_81','door_83','door_87','door_88','door_89','door_93'],'forbidden_doors': [],
                                                "map_size": 600,
                                                "cab_spawns":{
                                            "0":[(0.0,0.0,1.55),(1,-6.77,-3.06,-6.4),(1,-0.8,0.568,-5.6)],\
                                            "1":[(0.0, 0.0, -1.55),(1,-0.467,0.71,-2.0),(1,-0.365,1.3,1.2)],\
                                            "2":[(0.0, 0.0, 0.0),(0,-7.0,-5.9,-2.7 )],\
                                            "3":[(0.0, 0.0, 3.1),(0, -2.7,-6.1,-3.6),(0,1.27,-0.65,1.08),(0,1.25,2.4,4.05)],\
                                            "4":[(0.0, 0.0, 0.0),(-1,-10.5 ,-0.91),(-1,-2.52 ,8.49),(-1,-3.12 ,5.38),(-1,-3.28 ,2.14)]}}


        self.map_settings['Ihlen_1_int'] = {'grid_offset': np.array([11.5, 8.5, 15.1]),
                                                'grid_spacing': np.array([self.grid_res, self.grid_res, 0.1]),'offset': 0,
                                                'object_dist': 2.2,'object_dist_cabinets':2.2, 'door_dist': 1.4, 'doors': ['door_86', 'door_91','door_99','door_103','door_108'],'forbidden_doors': ['door_86', 'door_91'],
                                                "map_size": 600,
                                                "cab_spawns":{
                                            "0":[(0.0,0.0,3.1),(0,4.6,7.68,10.2),(0,-1.4,8.6,10.5)],\
                                            "1":[(0.0, 0.0, 0.0),(0,-4.64,7.88,10.25),(0,-4.58,0.64,2.87)],
                                            "2":[(0.0, 0.0, -1.55),(1,-4.53,-1.82,10.5),(1,-3.28,-1.50,5.82),(1,1.74,3.28,6.07)],
                                            "3":[(0.0, 0.0, 1.55),(1,-4.64,-2.44,7.8),(1,-4.6,-1.44,0.63),(1,-0.17,4.42,-1.0),(1,1.96,4.26,3.2)]}}



        # Remove some of the door for physically realistic simulated opening of doors. These might block pathways or narrow corridors 
        # Doors can potentially be opened but the procedure used in this project was simply not extended to it.
        self.map_settings['Beechwood_0_int']['doors'].remove('door_101')
        self.map_settings['Beechwood_0_int']['doors'].remove('door_97')
        if self.physically_simulated_openings:
            self.map_settings['Merom_1_int']['doors'].remove('door_88')
            self.map_settings['Merom_1_int']['doors'].remove('door_87')
import logging

import numpy as np
import pybullet as p


from igibson.utils.utils import l2_distance, restoreState
from igibson.objects.ycb_object import YCBObject
from igibson.utils.utils import cartesian_to_polar, l2_distance, rotate_vector_3d
import os
import igibson
from src.igibson.reward_functions.point_goal_REW import PointGoalReward
from src.igibson.reward_functions.generic_penalty_reward import GenericPenaltyReward
from src.igibson.reward_functions.cabinet_REW import CabinetReward
from src.igibson.reward_functions.out_of_bound_REW import OutOfBoundRew
from src.igibson.termination_conditions.point_goal_TER import PointGoal
from src.igibson.termination_conditions.timeout import Timeout
from src.igibson.termination_conditions.out_of_bound_TER import OutOfBound
from src.igibson.termination_conditions.max_collision import MaxCollision
from igibson.objects.articulated_object import URDFObject
from igibson import object_states
from src.utils.utils import get_all_openable_joints,close_all_drawers,hacky_cabinet_trick
from scipy.spatial.transform import Rotation as R
from itertools import permutations 
import copy

class HRLTask():

    def __init__(self, env,config):
        self.config = config

        self.target_dist_min = self.config.get("target_dist_min", 1.0)
        self.target_dist_max = self.config.get("target_dist_max", 10.0)
        self.num_tar_objects = self.config.get("tar_objects", 6.0)
        self.polar_to_geodesic = self.config.get("polar_to_geodesic", False)
        self.num_cabinets = self.config.get('num_cabinets', 3)
        self.num_cracker = self.config.get('num_cracker', 3)
        self.replace_objects = self.config.get("replace_objects", True)

        self.physically_simulated_openings = self.config.get("physically_simulated_openings", False)

        self.termination_conditions = [
            Timeout(self.config),
            OutOfBound(self.config),
            PointGoal(self.config),
        ]

        self.reward_functions = [
            PointGoalReward(self.config),
            OutOfBoundRew(self.config),
            GenericPenaltyReward(self.config),
        ]


        self.current_target_ind = 0
        self.prev_target_ind = 0
        self.num_categories = self.config.get("sem_categories", 1)
        self.queue_sample = False
        self.queue = []

        if self.config.get("scene_id",'Benevolence_0_int') in ["Benevolence_0_int","Rs_int"]:
            self.object_distance = 0.85
            self.cabinet_distance = 1.6
        else:
            self.object_distance = 2.0
            self.cabinet_distance = 2.0

        
        self.object_distance_cabinets = env.mapping.map_settings[env.last_scene_id]['object_dist_cabinets']
        
       


    def reset_cracker_objects(self, env):
        """
        Reset the poses of interactive objects to have no collisions with the scene or the robot

        :param env: environment instance
        """
        
        if(self.queue_sample):

            orn = np.array([0, 1, 1.5])
            for i,obj_list in enumerate(self.interactive_objects):
                state_id = p.saveState()
                pos = self.target_pos_list_cracker[i+3]
                self.target_pos_list.append(pos)
                #obj_list[0].force_wakeup()
                env.land(obj_list[0], pos, orn)
                
                pos2 = pos.copy()
                pos2[0] += 0.195
                env.land(obj_list[1], pos2, orn)
                
                pos3 = pos.copy()
                pos3[0] -= 0.195
                env.land(obj_list[2], pos3, orn)
                p.removeState(state_id)
            return True
        

        max_trials = 200
        
        for obj_list in self.interactive_objects:
            # TODO: p.saveState takes a few seconds, need to speed up
            state_id = p.saveState()
            for _ in range(max_trials):
                _, pos = env.scene.get_random_point(floor=self.floor_num)
                
                enough_distance_to_other_objects = True
                for already_pos in self.target_pos_list[-3::]:
                    dist = l2_distance(pos, already_pos)
                    if dist < self.object_distance:
                        enough_distance_to_other_objects = False
                        break
                if not enough_distance_to_other_objects:
                    continue
                
                enough_distance_to_doors = True
                for d_pos in self.door_pos_list:
                    if l2_distance(pos[:2],d_pos[:2]) < 1.4:
                        enough_distance_to_doors = False
                        break

                if not enough_distance_to_doors:
                    continue

                enough_distance_to_cabinets= True
                for already_pos in self.target_pos_list[:3]:
                    dist = l2_distance(pos, already_pos)
                    if dist < self.object_distance_cabinets:
                        enough_distance_to_cabinets = False
                        break
                if not enough_distance_to_cabinets:
                    continue

                
                if env.scene.build_graph:
                    _, dist = env.scene.get_shortest_path(
                    self.floor_num, self.initial_pos[:2], pos[:2], entire_path=False
                    )
                else:
                    dist = l2_distance(initial_pos, target_pos)
                
                orn = np.array([0, 1, 1.5])
                reset_success1 = env.test_valid_position(obj_list[0], pos, orn)

                pos2 = pos.copy()
                pos2[0] += 0.195
                
                reset_success2 = env.test_valid_position(obj_list[1], pos2, orn)
                
                pos3 = pos.copy()
                pos3[0] -= 0.195
                reset_success3 = env.test_valid_position(obj_list[2], pos3, orn)
                restoreState(state_id)
                reset_success = reset_success1 and reset_success2 and reset_success3
                if reset_success and (self.target_dist_min < dist):
                    break
                    

            
            if not reset_success:
                print("WARNING: Failed to reset interactive obj without collision")
                p.removeState(state_id)
                
                return False

            self.target_pos_list.append(pos)
            env.land(obj_list[0], pos, orn)
            env.land(obj_list[1], pos2, orn)
            env.land(obj_list[2], pos3, orn)
            p.removeState(state_id)
        return True

    
    def test_coll_with_cracker(self,env, obj, pos, orn=None, ignore_self_collision=True):
        

        env.set_pos_orn_with_z_offset(obj, pos, orn)


        ignore_ids = obj.get_body_ids() if ignore_self_collision else []
        env.simulator.step()
        for body_id in obj.get_body_ids():
            collisions = [x for x in p.getContactPoints(bodyA=body_id) if x[2] not in ignore_ids]
            if len(collisions) > 0:
                break
        
        return len(collisions) > 0
    
    def put_plate_in_drawer(self,plate,cabinet):
        not_in_correct_drawer = True
        while not_in_correct_drawer:
            try:
                assert plate.states[object_states.Inside].set_value(cabinet, True, use_ray_casting_method=True)
                p.changeDynamics(plate.get_body_ids()[0], -1,
                                 activationState=p.ACTIVATION_STATE_WAKE_UP)
                p.changeDynamics(cabinet.get_body_ids()[0], -1,
                                 activationState=p.ACTIVATION_STATE_WAKE_UP)
                
            except:
                
                return False

                
            plate_pos = round(plate.get_position()[2],3)
            if plate_pos > 0.5 and plate_pos < 0.6:
                not_in_correct_drawer = False

    def reset_cabin_objects_pre_defined(self,env,cabinet,plate,ind):
        
        state_id = p.saveState()
        spawns = env.mapping.map_settings[env.last_scene_id]['cab_spawns']
        spawn_area_ind = str(np.random.choice(np.arange(len(spawns.keys()))))

        initial_orn = spawns[spawn_area_ind][0]
        #first index is always the orientation of the cabinets in the corresponding spawn area!
        pos_ind = np.random.choice(np.arange(len(spawns[spawn_area_ind])-1)) + 1
        position = spawns[spawn_area_ind][pos_ind] 
        axis = position[0]
        
        if axis == 0:
            low = position[2]
            high=position[3]
            initial_pos = np.array([position[1], np.random.uniform(low=low, high=high), 0.0])
        elif axis == 1:
            low = position[1]
            high=position[2]
            initial_pos = np.array([np.random.uniform(low=low, high=high),position[3], 0.0])

        elif axis == -1:
            initial_pos = np.array([position[1],position[2], 0.0])

        
        for other_cabinets in self.target_pos_list:
            if l2_distance(other_cabinets[:2],initial_pos[:2]) < self.cabinet_distance:
                p.removeState(state_id)
                return False

        cabinet.force_wakeup()
        has_collision = self.test_coll_with_cracker(env,cabinet, initial_pos, initial_orn,ignore_self_collision=True)
        
        restoreState(state_id)

        if has_collision:
            p.removeState(state_id)
            return False
        
        env.land(cabinet,initial_pos,initial_orn)

        if int(ind) in self.cabinet_target_dict:

            plate.set_base_link_position_orientation([-75, -100, -100], [0, 0, 0, 1])
            p.changeDynamics(plate.get_body_ids()[0], -1,
                             activationState=p.ACTIVATION_STATE_WAKE_UP)
            if env.physically_simulated_openings:
                self.put_plate_in_drawer(plate,cabinet)
            
            self.plate_pos.append(0)
        else:
            plate.set_base_link_position_orientation([-75, -100, -100], [0, 0, 0, 1])
            p.changeDynamics(plate.get_body_ids()[0], -1,
                             activationState=p.ACTIVATION_STATE_WAKE_UP)
            #for cabinets with no plates, open just the first drawer which will be empty.
            self.plate_pos.append(0)

        self.target_pos_list.append(initial_pos)
        self.target_pos_orient_list.append(initial_orn[-1])
        p.removeState(state_id)

        #essentially fixing all the other joints such that the do not randomly open
        hacky_cabinet_trick(cabinet.get_body_ids()[0],[self.all_cabinet_joints[0]]+self.all_cabinet_joints[2::])
        return True
        

    


    def load_cabinet_objects(self,env):
        print("Load Objects")
        cabinet_scale = np.array([0.65, 0.9,0.65])

        bounding_box_scale = [0.2, 0.2, 0.02]
        
        
        self.cabinets = []
        self.microwaves = []
        self.plates = []
        self.grasp_orientation = {"-1.55":([-0.00, 0.999, -0.04] ,[0.0,-0.280,0.556],np.array([-0.69094545 ,-0.00988954 , 0.72243476 ,-0.02417938])),"1.55":([-0.00, -0.999, -0.04] ,[0.0,0.280,0.556],np.array([-0.69094545 ,-0.00988954 , 0.72243476 ,-0.02417938]))\
        ,"0.0":([-0.99, 0.009, -0.04] ,[0.280,0.0,0.556],np.array([-0.47764134, -0.51253116,  0.51409191 ,-0.49485371])),"3.1":([0.99, 0.009, -0.04] ,[-0.280,0.0,0.556],np.array([-0.47764134 ,-0.51253116 , 0.51409191, -0.49485371]))}
        
        self.pos_offset_agent_to_cabinets = {"-1.55":np.array([0.0,-0.85,0.0]),"1.55":np.array([0.0,0.85,0.0]),"0.0":np.array([0.85,0.0,0.0]),"3.1":np.array([-0.85,0.0,0.0])}

        cabinet_filename = os.path.join(igibson.assets_path, "models/cabinet2/cabinet_0007.urdf")

        microwave_dir = os.path.join(igibson.ig_dataset_path, "objects/microwave/7128/")
        microwave_filename = os.path.join(microwave_dir, "7128.urdf")

        # Load cabinet, set position manually, and step simulate 100 timesteps
        self.cabinet_ids = [91,92,93,94,95,96]

        #Important: project Cabin and plate to the same color in order to match the obs-space for exploration policy
        self.cabinet_cat_to_ind = {91:0,92:1,93:2,94:3,95:4,96:5}
        self.ignore_plate_ids = []
        self.plate_ids = [201,202,203]
        self.plate_cat_to_ind = {201:0,202:1,203:2}
        for i in range(self.num_cabinets):

            kwargs = {"class_id":91+i}
            cabinet = URDFObject(filename=cabinet_filename, name="cab_{}".format(i),category="cabinet", scale=cabinet_scale,**kwargs)
            env.simulator.import_object(cabinet)
            
            cabinet.set_position([-np.random.uniform(50,100),-np.random.uniform(50,100),-np.random.uniform(50,100)])#set_base_link_position_orientation([-np.random.uniform(50,100),-np.random.uniform(50,100),-np.random.uniform(50,100)],[0,0,0,1])
    
            # Load plate, set position on top of the cabinet, open it, and step 100 times
            kwargs = {"class_id":201+i}

            plate_dir = os.path.join(igibson.ig_dataset_path, "objects/plate/plate_000/")
            plate_filename = os.path.join(plate_dir, "plate_000.urdf")

            plate = URDFObject(filename=plate_filename, name="plate_{}".format(i),category="plate", model_path=plate_dir, bounding_box=bounding_box_scale,**kwargs)
        
            env.simulator.import_object(plate)
        
            self.cabinets.append(cabinet)
            self.plates.append(plate)
            self.already_opened["cab_{}".format(i)] = False

            self.ignore_plate_ids.append([plate.get_body_ids()[0]])
            self.all_cabinet_joints = get_all_openable_joints(cabinet.get_body_ids()[0])

        
    def load_cracker_objects(self, env):
        """
        Load interactive objects (YCB objects)

        :param env: environment instance
        :return: a list of interactive objects
        """
        self.interactive_objects = []
        object_paths = []

        self.cracker_ids = np.arange(80,80+self.num_cracker)
        self.cats_to_ind = {}
        for _ in range(self.num_cracker):
            object_paths.append('003_cracker_box')

        object_ids = []
        for i in range(self.num_cracker):
            object_ids.append(80+i)
            self.cats_to_ind[80+i] = i+3

       
        self.remove_collision_links = []
        self.target_pos_list = []
        self.target_pos_orient_list = []
        
        for i in range(self.num_cracker):
            kwargs = {"class_id":object_ids[i]}
            obj1 = YCBObject(name=object_paths[i],**kwargs)
            obj2 = YCBObject(name=object_paths[i],**kwargs)
            obj3 = YCBObject(name=object_paths[i],**kwargs)

            env.simulator.import_object(obj1)
            env.simulator.import_object(obj2)
            env.simulator.import_object(obj3)
            self.interactive_objects.append([obj1,obj2,obj3])

            self.remove_collision_links.append(obj1.get_body_ids()[0])
            self.remove_collision_links.append(obj2.get_body_ids()[0])
            self.remove_collision_links.append(obj3.get_body_ids()[0])

        print("COLLISIONS LINKS:",self.remove_collision_links)
   
    def load_door_material(self,env):

        #load door material 
        self.door_sem_ids = np.arange(375, 390)
        self.forbidden_door_sem_ids = []

        self.door_cat_to_ind = {}
    
        keep_door_list = env.mapping.map_settings[env.last_scene_id]['doors']
        forbidden_door_list = env.mapping.map_settings[env.last_scene_id]['forbidden_doors']
        self.forbidden_door_ids = []
        self.all_door_ids = []
        self.door_dict = {}
        self.door_pos_list = []
        self.door_key_to_index = {}
        self.door_index_to_key = {}
        self.already_opened = {}
        self.pos_offset_axis_agent_to_doors = {"3.0":1,"-3.0":1,"0.0":1,"2.0":0,"-2.0":0, "-1.0":0,"1.0":0,"-0.0":1,"0.0":1}
        
        self.offset_positions_door_knop = {
        "Rs_int":
        {"door_52":(np.array([-2.32955889912822777, 1.5896568178916145, 1.1660137611763965]),[0.115, -0.99, -0.04],np.array([ 0.47259939, -0.51649904, -0.45916283,  0.54685283]))},\
        "Merom_0_int":{
        "door_63":(np.array((1.167703034849319, 8.00487977734645, 1.1731034189127612)),[0.99, 0.115, -0.04],np.array([0.72884667, 0.01690894, 0.02199815 ,0.68411452])),\
        "door_64":(np.array((2.47241509103686, 4.121037815728697, 1.1738688381864746)),[0.115, -0.99, -0.04],np.array([ 0.47259939, -0.51649904, -0.45916283,  0.54685283])),\
        "door_67":(np.array((3.860885040271105, 3.038969153757917, 1.1755618255840607)),[0.115, 0.99, -0.04],np.array([0.48231643, 0.50334662 ,0.51021588 ,0.50367928]))
        },
        "Benevolence_0_int":
        {"door_11":(np.array((-2.423500850138208, -4.7210416622779166, 1.174230708550517)),[0.115, 0.99, -0.04],np.array([0.48231643, 0.50334662 ,0.51021588 ,0.50367928]))}
        ,
        "Pomaria_0_int":
        {
        "door_40":(np.array((-6.301048439055762, 1.938020511760675, 1.2924449026775486)),[0.99, 0.115, -0.04],np.array([0.72884667, 0.01690894, 0.02199815 ,0.68411452])),
        "door_44":(np.array((-2.7488139063471166, 1.3682661727658787, 1.289630668518407)),[0.115, 0.99, -0.04],np.array([0.48231643, 0.50334662 ,0.51021588 ,0.50367928])),
        "door_46":(np.array((-2.260067239074148, 2.461672187966188, 1.2882166672553788)),[0.115, -0.99, -0.04],np.array([ 0.47259939, -0.51649904, -0.45916283,  0.54685283]))
        },
        "Wainscott_1_int":
        {
        "door_86":(np.array((3.335803871606912, 4.8975472137415235, 1.16287336352032)),[0.115, -0.99, -0.04],np.array([ 0.47259939, -0.51649904, -0.45916283,  0.54685283]),"special_offset"),
        "door_88":(np.array((1.2604323142487202, 5.481701979654908, 1.168560782767373)),[0.115, -0.99, -0.04],np.array([ 0.47259939, -0.51649904, -0.45916283,  0.54685283])),
        "door_89":(np.array((-1.3400364303922971, 4.976185952912902, 1.1670853725490626)),[0.99, 0.115, -0.04],np.array([0.72884667, 0.01690894, 0.02199815 ,0.68411452])),
        "door_95":(np.array((-0.30671819975590847, 7.831625398243174, 1.1654201109186402)),[0.115, -0.99, -0.04],np.array([ 0.47259939, -0.51649904, -0.45916283,  0.54685283]))
        },
        "Ihlen_0_int":
        {
        "door_46":(np.array((-0.47224390457252663, 7.344832904318937, 1.1640381312343198)),[0.115, 0.99, -0.04],np.array([0.48231643, 0.50334662 ,0.51021588 ,0.50367928])),
        "door_47":(np.array((2.6054329096784454, 5.3814463857009756, 1.1686194994252783)),[0.15, -0.99, -0.04],np.array([ 0.47259939, -0.51649904, -0.45916283,  0.54685283]),"special_offset_2"),
        },
        
        "Beechwood_1_int":
        {
        "door_80":(np.array((-1.505750294721664, 2.0010695755288532, 1.2884099875754214)),[-0.99, 0.115, -0.04],np.array([0.00857348 ,0.73057497, 0.68255419, 0.01750836])),
        "door_81":(np.array((-1.5057601284356021, 0.8930904720205075, 1.2873128640133114)),[-0.99, 0.115, -0.04],np.array([0.00857348 ,0.73057497, 0.68255419, 0.01750836])),
        "door_87":(np.array((-2.3100598366318085, -2.254764976759522, 1.2856019592008054)),[-0.99, 0.115, -0.04],np.array([0.00857348 ,0.73057497, 0.68255419, 0.01750836])),
        "door_83":(np.array((-7.460034790489249, -1.7164235578917244, 1.2910582985444077)),[0.99, 0.115, -0.04],np.array([0.72884667, 0.01690894, 0.02199815 ,0.68411452])),
        },
        "Ihlen_1_int":
        {
        "door_103":(np.array((0.2623856577922199, 4.335419564910796, 1.1699523261622826)),[-0.99, 0.115, -0.04],np.array([0.00857348 ,0.73057497, 0.68255419, 0.01750836]),"special_offset_3"),
        "door_99":(np.array((1.2914351844914307, 7.595435491068094, 1.1734937371929233)),[-0.99, 0.115, -0.04],np.array([0.00857348 ,0.73057497, 0.68255419, 0.01750836])),
        "door_108":(np.array((3.952256680465771, 6.4398975481115865, 1.17222043045139)),[0.115, 0.99, -0.04],np.array([0.48231643, 0.50334662 ,0.51021588 ,0.50367928])),
        },

        "Pomaria_2_int":
        {
        
        "door_32":(np.array((-2.429874814125906, 2.912410901468633, 1.2922978245009666)),[0.99, 0.115, -0.04],np.array([0.72884667, 0.01690894, 0.02199815 ,0.68411452]),'special_offset_5'),
        "door_29":(np.array((-4.108146957972831, 1.3011289610536825, 1.2909079212648036)),[-0.99, 0.115, -0.04],np.array([0.00857348 ,0.73057497, 0.68255419, 0.01750836])),
        
        },
        
        "Benevolence_2_int":
        {
        
        "door_41":(np.array((-2.3294251298410047, -5.7318323560171, 1.1756318442424931)),[0.115, 0.99, -0.04],np.array([0.48231643, 0.50334662 ,0.51021588 ,0.50367928]),"special_offset_3"),
        "door_43":(np.array((-0.07457950824226703, -4.03896250440451, 1.173443335054869)),[0.115, -0.99, -0.04],np.array([ 0.47259939, -0.51649904, -0.45916283,  0.54685283])),
        "door_35":(np.array((-1.658916999104231, 0.6433604714979888, 1.1776021577939366)),[0.99, 0.115, -0.04],np.array([0.72884667, 0.01690894, 0.02199815 ,0.68411452])),
        },
        
        
        "Benevolence_1_int":
        {
        
        "door_54":(np.array((-1.986552772364997, -0.3250635332883638, 1.1779291644494607)),[-0.99, 0.115, -0.04],np.array([0.00857348 ,0.73057497, 0.68255419, 0.01750836])),
        "door_55":(np.array((-3.462870694289125, 0.28348957866678615, 1.1714520618014208)),[0.115, 0.99, -0.04],np.array([0.48231643, 0.50334662 ,0.51021588 ,0.50367928]),"special_offset_3"),
        
        },
        "Wainscott_0_int":
        {
        
        "door_137":(np.array((-3.2700937176536993, 0.010373533327213191, 1.1701871594153512)),[0.115, -0.99, -0.04],np.array([ 0.47259939, -0.51649904, -0.45916283,  0.54685283])),
        "door_136":(np.array((-1.9119653861183323, -0.11104122701499752, 1.1723935258658593)),[0.115, 0.99, -0.04],np.array([0.48231643, 0.50334662 ,0.51021588 ,0.50367928])),
       
        "door_134":(np.array((1.6507323245824974, 9.055452417892136, 1.1678363339361633)),[0.115, 0.99, -0.04],np.array([0.48231643, 0.50334662 ,0.51021588 ,0.50367928])),
        },
        "Beechwood_0_int":
        {
       
        "door_101":(np.array((0.2623856577922199, 4.335419564910796, 1.1699523261622826)),[0.115, 0.99, -0.04],np.array([0.48231643, 0.50334662 ,0.51021588 ,0.50367928])),
        "door_97":(np.array((-1.8876932289081214, 0.7844100528201347, 1.2890816901399837)),[0.515, -0.99, -0.04],np.array([ 0.47259939, -0.51649904, -0.45916283,  0.54685283]),"special_offset_4"),
        
        "door_98":(np.array((-1.2219464968971454, 1.900459965723059, 1.2895142375646662)),[0.115, -0.99, -0.04],np.array([ 0.47259939, -0.51649904, -0.45916283,  0.54685283])),
        
        "door_102":(np.array((-4.169451767612317, -0.5724247078982116, 1.28770782734075)),[0.99, 0.115, -0.04],np.array([0.72884667, 0.01690894, 0.02199815 ,0.68411452])),
        },

        "Merom_1_int":
        {
        
        "door_85":(np.array((1.560086484568731, 2.258955463391304, 1.1709400771018303)),[0.115, 0.99, -0.04],np.array([0.48231643, 0.50334662 ,0.51021588 ,0.50367928]),"special_offset_3"),
        "door_86":(np.array((0.4281887231684964, 2.2580139427157673, 1.171085573283589)),[0.115, 0.99, -0.04],np.array([0.48231643, 0.50334662 ,0.51021588 ,0.50367928]),"special_offset_3"),
        "door_89":(np.array((0.4281887231684964, 2.2580139427157673, 1.171085573283589)),[0.115, 0.99, -0.04],np.array([0.48231643, 0.50334662 ,0.51021588 ,0.50367928]),"special_offset_3"),
        
        "door_88":(np.array((3.952256680465771, 6.4398975481115865, 1.17222043045139)),[0.115, -0.99, -0.04],np.array([ 0.47259939, -0.51649904, -0.45916283,  0.54685283])),
        "door_87":(np.array((3.952256680465771, 6.4398975481115865, 1.17222043045139)),[0.115, -0.99, -0.04],np.array([ 0.47259939, -0.51649904, -0.45916283,  0.54685283])),
        },
        "Pomaria_1_int":
        {
        "door_73":(np.array((-4.438230524337128, 2.0820288290884235, 1.2866354371696536)),[0.115, -0.99, -0.04],np.array([ 0.47259939, -0.51649904, -0.45916283,  0.54685283]),"special_offset_3"),
        "door_72":(np.array((-3.5114173382965563, 2.081037616758743, 1.2917324725038868)),[0.115, -0.99, -0.04],np.array([ 0.47259939, -0.51649904, -0.45916283,  0.54685283])),
        
        },


        }
        
        
        ind = 0
        for key in env.scene.door_list.keys():
            door_id = env.scene.door_list[key][0].get_body_ids()[0]
            door_pos_orient = p.getBasePositionAndOrientation(env.scene.door_list[key][0].get_body_ids()[0])
            pos_door = door_pos_orient[0]
            orient_door = p.getEulerFromQuaternion(door_pos_orient[1])
            
            if key in forbidden_door_list:
                self.forbidden_door_ids.append(door_id)
                self.forbidden_door_sem_ids.append(env.scene.door_list[key][1])
                

            if key not in keep_door_list or (ind > 3 and key not in forbidden_door_list):
               
                env.scene.remove_object(env.scene.door_list[key][0])
                
                env.scene.door_list[key][0].main_body_is_fixed = False
                env.scene.door_list[key][0].set_base_link_position_orientation([-np.random.uniform(50,100),-np.random.uniform(50,100),-np.random.uniform(50,100)],[0,0,0,1])

                
            else:
                if key not in forbidden_door_list:
                    self.door_pos_list.append(pos_door)
                    self.door_dict[key] = [door_id,pos_door,np.round(orient_door[-1])]

                    self.all_door_ids.append(door_id)
                    self.door_key_to_index[key] = ind
                    self.door_index_to_key[ind] = key
                    self.already_opened[key] = False
                    #this extracts the semantic category which has been set in the scene loading process used by the mapping module.
                    self.door_cat_to_ind[env.scene.door_list[key][1]] = ind
                    ind += 1
                
        
        self.original_door_mapping = self.door_cat_to_ind.copy()

    def reset_doors(self,env):
        #opens all doors, even the ones leading to the cliff (end of map)

        for door_k in self.door_dict.keys():

            door_id = self.door_dict[door_k][0]
            env.scene.open_one_obj(door_id,mode="zero") #if you want the colorized doors to be open in the beginning, put mode="max"
            if door_id in self.forbidden_door_ids:
                pairs = env.scene.open_one_obj(door_id,mode="zero")
                
            else:
                if not self.physically_simulated_openings:
            
                    for joint_id in range(p.getNumJoints(door_id)):
                        p.changeDynamics(door_id,joint_id, mass=9999999.0,lateralFriction=0.1,spinningFriction=0.1,rollingFriction=0.1,frictionAnchor=True)

                
                p.changeDynamics(door_id, -1, activationState=p.ACTIVATION_STATE_WAKE_UP)

    
    def get_task_obs(self, env):
        """
        Get task-specific observation, including goal position, current velocities, etc.

        :param env: environment instance
        :return: task-specific observation
        """
        

        # linear velocity along the x-axis
        linear_velocity = rotate_vector_3d(env.robots[0].get_linear_velocity(), *env.robots[0].get_rpy())[0]
        # angular velocity along the z-axis
        angular_velocity = rotate_vector_3d(env.robots[0].get_angular_velocity(), *env.robots[0].get_rpy())[2]
       
        return linear_velocity, angular_velocity
    
    def sample_initial_pose_and_target_pos(self, env):
        """
        Sample robot initial pose and target position

        :param env: environment instance
        :return: initial pose and target position
        """
        
        self.queue_sample = False

        if ((np.random.uniform() < 0.12 and len(env.queue_for_task_resampling) > 0)):

            for i in range(len(env.queue_for_task_resampling)):
                initial_pos, initial_orn, tar_pos_list, init_obs, scene_id = env.queue_for_task_resampling.pop(0)
                # check whether current scene id is the one used in sample
                if env.last_scene_id == scene_id:
                    self.init_obs = init_obs
                    self.target_pos_list_cracker = tar_pos_list
                    self.queue_sample = True
                    
                    return initial_pos, initial_orn
                env.queue_for_task_resampling.append((initial_pos, initial_orn, tar_pos_list, init_obs, scene_id))
        if self.queue_sample:
            return initial_pos,initial_orn
        
        
        self.queue_sample = False
        _, initial_pos = env.scene.get_random_point(floor=self.floor_num)

        initial_orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])

        return initial_pos, initial_orn

    def reset_scene(self, env):
        """
        Task-specific scene reset: get a random floor number first

        :param env: environment instance
        """
        self.floor_num = env.scene.get_random_floor()
        
        env.scene.reset_scene_objects()
        self.reset_doors(env)


    def sample_target_object(self):

        self.wanted_objects = np.zeros(self.num_categories)
        self.indices = np.random.choice(np.arange(self.num_categories),size=self.num_tar_objects,replace=self.replace_objects)

        
        self.wanted_objects[self.indices] = 1.0
        self.num_cat_in_episode = (np.unique(self.indices)).shape[0]
        self.uniq_indices = np.unique(self.indices)
        
        self.initial_wanted = self.wanted_objects.copy()

        if(self.queue_sample):
            self.wanted_objects = self.init_obs
            self.uniq_indices = np.argwhere(self.wanted_objects != 0)[:,0]



    def get_reward(self, env, collision_links=[], action=None, info={}):

        reward = 0.0
        for reward_function in self.reward_functions:
            reward += reward_function.get_reward(self, env,info)

        return reward, info

    def reset_variables(self, env):
        self.path_length = 0.0
        self.robot_pos = self.initial_pos[:2]

    def reset(self,env):
        self.reset_scene(env)
        succ = self.reset_agent(env)
        self.reset_variables(env)

        for reward_function in self.reward_functions:
            reward_function.reset(self, env)
        for termination_condition in self.termination_conditions:
            termination_condition.reset(self, env)

        return succ

    def get_termination(self, env, collision_links=[], action=None, info={}):

        done = False
        success = False
        for condition in self.termination_conditions:
            d, s = condition.get_termination(self, env)
            done = done or d
            success = success or s
        info["done"] = done
        info["success"] = success
        return done, info

    def global_to_local(self, env, pos):
        """
        Convert a 3D point in global frame to agent's local frame

        :param env: environment instance
        :param pos: a 3D point in global frame
        :return: the same 3D point in agent's local frame
        """
        return rotate_vector_3d(np.array(pos) - np.array(env.robots[0].get_position()), *env.robots[0].get_rpy())

    def local_to_global(self,env,pos):
        local_to_global = R.from_euler("xyz", [*env.robots[0].get_rpy()]).as_matrix()

        return np.dot(local_to_global, np.array(pos) ) + np.array(env.robots[0].get_position())

    
    def step(self, env):
        pass

    def get_global_geodesic_length_naive(self,env):
        #put all undirected nodes connected to each other into a list of tuples
        #this has been done due to genetric optimization but in here we
        #gonna provide the global optimal solution since we run through all permutations
        #of possible driving scenarios.
        dist_list = []
        source = self.initial_pos[:2]
        pos_list = self.target_pos_list
        pairs = []
        
        for i in range(self.num_categories):
            if self.wanted_objects[i] == 1:
                waypoints, geodesic_dist = env.scene.get_shortest_path(self.floor_num, source, self.target_pos_list[i][:2],
                                                                               entire_path=True)
                dist_list.append((0, i+1, geodesic_dist)) #[0,ind]
                pairs.append((0,i+1))
                

        for i in range(1,self.num_categories+1):
            if self.wanted_objects[i-1] == 1:
                source = self.target_pos_list[i-1][:2]
                for j in range(1,self.num_categories+1):
                    target = self.target_pos_list[j-1][:2]
                    if self.wanted_objects[j-1] == 1 and i != j and (i,j) not in pairs and (j,i) not in pairs:
                        waypoints, geodesic_dist = env.scene.get_shortest_path(self.floor_num, source, target,
                                                                               entire_path=True)
                        dist_list.append((i,j,geodesic_dist))
                        pairs.append((i,j))




        listi = [i+1 for i in np.argwhere(self.wanted_objects==1)[:,0]]
        
        all_permutations = permutations(listi)
        c_permuts = list(all_permutations).copy()
        
        distances = []

        for ind_p in c_permuts:
            d_ = 0
            for i,p in enumerate(ind_p):
                node = p 
                if i == 0:
                    dist_ind = pairs.index((0,node))
                    dist = dist_list[dist_ind][2]
                    d_ += dist
                else:
                    if (ind_p[i-1],node) in pairs:
                        dist_ind = pairs.index((ind_p[i-1],node))
                        dist = dist_list[dist_ind][2]
                        d_ += dist
                    else:
                        dist_ind = pairs.index((node,ind_p[i-1]))
                        dist = dist_list[dist_ind][2]
                        d_ += dist
            distances.append(d_)

        min_dist_ind = np.argmin(np.array(distances))

        self.initial_geodesic_length = distances[min_dist_ind]  



    def get_global_geodesic_length(self,env):
        n = (self.wanted_objects == 1.0).sum()+1
        
        dist_list = []
        source = self.initial_pos[:2]
        pos_list = self.target_pos_list

        ind = 1
        for i in range(self.num_categories):
            if self.wanted_objects[i] == 1:
                waypoints, geodesic_dist = env.scene.get_shortest_path(self.floor_num, source, self.target_pos_list[i][:2],
                                                                               entire_path=True)
                dist_list.append((0, ind, geodesic_dist)) #[0,ind]
                ind += 1

        pairs = []
        for i in range(1,self.num_categories+1):
            if self.wanted_objects[i-1] == 1:
                source = self.target_pos_list[i-1][:2]
                for j in range(1,self.num_categories+1):
                    target = self.target_pos_list[j-1][:2]
                    if self.wanted_objects[j-1] == 1 and i != j and (i,j) not in pairs and (j,i) not in pairs:
                        waypoints, geodesic_dist = env.scene.get_shortest_path(self.floor_num, source, target,
                                                                               entire_path=True)
                        dist_list.append((i,j,geodesic_dist))
                        pairs.append((i,j))
          
        print("DISTLIST:",dist_list)  
        # Initialize fitness function object using dist_list
        fitness_dists = mlrose.TravellingSales(distances = dist_list)
        problem_fit = mlrose.TSPOpt(length = n, fitness_fn = fitness_dists, maximize=False)

        best_state, best_fitness = mlrose.genetic_alg(problem_fit,  max_attempts = 100,random_state = 2)

        source = self.initial_pos[:2]
        target = self.target_pos_list[best_state[-2]-1][:2]
        waypoints, geodesic_dist = env.scene.get_shortest_path(self.floor_num, source, target,
                                                                               entire_path=True)
        best_fitness -= geodesic_dist
        
    def get_initial_geodesic_length(self, env):
        # First find neasrest Object
        all_distances = []
        source = self.initial_pos[:2]
        already_visited = []
        while True:
            geodesic_dists = [np.inf] * self.num_categories
            waypoints_list = [[]] * self.num_categories
            base_case = True
            for i in range(self.num_categories):
                if i in already_visited:
                    continue
                target = self.target_pos_list[i][:2]
                if self.wanted_objects[i] == 1:
                    waypoints, geodesic_dist = env.scene.get_shortest_path(self.floor_num, source, target,
                                                                           entire_path=True)
                    waypoints_list[i] = waypoints
                    geodesic_dists[i] = geodesic_dist
                    base_case = False

            if base_case:
                break
            nearest_object_from_source_ind = np.argmin(geodesic_dists)
            all_distances.append(geodesic_dists[nearest_object_from_source_ind])
            already_visited.append(nearest_object_from_source_ind)
            source = self.target_pos_list[i][:2]

        self.initial_geodesic_length = sum(all_distances)




    def reset_agent(self, env):
        """
        Reset robot initial pose.
        Sample initial pose and target position, check validity, and land it.

        :param env: environment instance
        """
        #close all drawers of cabinet
        
        for i in range(self.num_cabinets):
            close_all_drawers(self.cabinets[i].get_body_ids()[0],self.all_cabinet_joints,self.ignore_plate_ids[i])

        

        reset_success = False
        max_trials = 150

        self.floor_num = env.scene.get_random_floor()
        spawn_procedure_succ = True
        
        self.target_pos_list = []
        self.target_pos_orient_list = []
        self.plate_pos = []
        self.opened_cabinet = []
        # cache pybullet state
        # TODO: p.saveState takes a few seconds, need to speed up
        state_id = p.saveState()
        for i in range(max_trials):
            initial_pos, initial_orn = self.sample_initial_pose_and_target_pos(env)

            reset_success = env.test_valid_position(env.robots[0], initial_pos, initial_orn)
            restoreState(state_id)
            if reset_success:
                break

        if not reset_success:
            logging.warning("WARNING: Failed to reset robot without collision")

        p.removeState(state_id)

        
        self.initial_pos = initial_pos
        self.initial_orn = initial_orn
        
        env.land(env.robots[0],self.initial_pos, self.initial_orn)
        #sample which cainets contain plates
        self.sample_target_object()

        """
        added for samplng which objects are in which cabinet
        """
        sampled_cabinet_objects = np.argwhere(self.wanted_objects[:3] != 0)
        cabinet_perm = np.random.permutation([0,1,2])
        self.cabinet_target_dict = {}
        for sco in range(sampled_cabinet_objects.shape[0]):
            self.cabinet_target_dict[cabinet_perm[sco]] = sampled_cabinet_objects[sco,0]

       
        for cracker_ob in self.interactive_objects:
            cracker_ob[0].set_base_link_position_orientation([-np.random.uniform(50, 100), -np.random.uniform(50, 100), -np.random.uniform(50, 100)], [0, 0, 0, 1])
            cracker_ob[1].set_base_link_position_orientation([-np.random.uniform(50, 100), -np.random.uniform(50, 100), -np.random.uniform(50, 100)], [0, 0, 0, 1])
            cracker_ob[2].set_base_link_position_orientation([-np.random.uniform(50, 100), -np.random.uniform(50, 100), -np.random.uniform(50, 100)], [0, 0, 0, 1])
        
        #spawn plates away
        for it in range(self.num_cabinets):
            self.plates[it].set_base_link_position_orientation([-np.random.uniform(50, 100), -np.random.uniform(50, 100), -np.random.uniform(95, 150)], [0, 0, 0, 1])

        succ = False
        for it in range(self.num_cabinets):
            for i in range(max_trials):

                succ = self.reset_cabin_objects_pre_defined(env, self.cabinets[it],
                                                            self.plates[it],it)  
                if succ:
                    break

        if not succ:
            spawn_procedure_succ = False
            return spawn_procedure_succ
        succ = False

        for i in range(max_trials):
            succ = self.reset_cracker_objects(env)
            
            if succ:
                break

        if not succ:
            spawn_procedure_succ = False
            return spawn_procedure_succ

        if len(self.target_pos_list) < 6:
            spawn_procedure_succ= False
            return spawn_procedure_succ

        
        for key in self.already_opened.keys():

            if key.startswith('cab'):
                
                env.scene.restore_object_states_single_object(env.scene.objects_by_name[key], env.scene.object_states[key])
                p.changeDynamics(env.scene.objects_by_name[key].get_body_ids()[0], -1, activationState=p.ACTIVATION_STATE_WAKE_UP)
            else:
                p.changeDynamics(env.scene.door_list[key][0].get_body_ids()[0], -1, activationState=p.ACTIVATION_STATE_WAKE_UP)
            self.already_opened[key] = False

        if env.evaluate:
            self.get_global_geodesic_length_naive(env)
        
        return spawn_procedure_succ
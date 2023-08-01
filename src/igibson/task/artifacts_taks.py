#Artifacts loading cabinets etc..

 """
        #cabinet opening stuff
        if self.wanted_objects[-1] == 0:
            self.wanted_objects[-1] = 1
            self.uniq_indices = np.append(self.uniq_indices,[5])
            self.initial_wanted = self.wanted_objects.copy()

"""
        
        
"""
        #reset the doors to their original position
        for key in self.door_dict.keys():
            #initial door positions and orientations
            env.scene.door_list[key].main_body_is_fixed = True
            pos = self.door_dict[key][1]
            orient = self.door_dict[key][2]
            env.scene.door_list[key].set_base_link_position_orientation(pos, orient)




        for key in self.already_opened.keys():
            p.changeDynamics(env.scene.door_list[key].get_body_ids()[0], -1, activationState=p.ACTIVATION_STATE_WAKE_UP)
            #env.scene.door_list[key].reset()
            self.already_opened[key] = False
"""
"""
        #---------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------------------------------
        kwargs = {"class_id": 93}
        cabinet2 = URDFObject(filename=cabinet_filename, name="cab2", category="cabinet",
                             scale=self.cabinet_scale, **kwargs)
        env.simulator.import_object(cabinet2)

        # cabinet.set_position([0,0,0.5])

        # Load microwave, set position on top of the cabinet, open it, and step 100 times
        kwargs = {"class_id": 94}
        microwave2 = URDFObject(
            filename=microwave_filename, name="mic2", category="microwave", model_path=microwave_dir,
            scale=self.microwave_scale, **kwargs
        )
        env.simulator.import_object(microwave2)
        cabinet2.set_base_link_position_orientation(
            [-np.random.uniform(50, 100), -np.random.uniform(50, 100), -np.random.uniform(50, 100)], [0, 0, 0, 1])
        microwave2.set_base_link_position_orientation(
            [-np.random.uniform(50, 100), -np.random.uniform(50, 100), -np.random.uniform(50, 100)], [0, 0, 0, 1])
        self.cabinets.append(cabinet2)
        self.microwaves.append(microwave2)
        print("Load Objects2")
        # ---------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------------------------------

        kwargs = {"class_id": 95}
        cabinet3 = URDFObject(filename=cabinet_filename, name="cab3", category="cabinet",
                             scale=self.cabinet_scale, **kwargs)
        env.simulator.import_object(cabinet3)

        # cabinet.set_position([0,0,0.5])

        # Load microwave, set position on top of the cabinet, open it, and step 100 times
        kwargs = {"class_id": 96}
        microwave3 = URDFObject(
            filename=microwave_filename, name="mic3", category="microwave", model_path=microwave_dir,
            scale=self.microwave_scale, **kwargs
        )
        env.simulator.import_object(microwave3)
        cabinet3.set_base_link_position_orientation(
            [-np.random.uniform(50, 100), -np.random.uniform(50, 100), -np.random.uniform(50, 100)], [0, 0, 0, 1])
        microwave3.set_base_link_position_orientation(
            [-np.random.uniform(50, 100), -np.random.uniform(50, 100), -np.random.uniform(50, 100)], [0, 0, 0, 1])
        self.cabinets.append(cabinet3)
        self.microwaves.append(microwave3)
        print("Load Objects3")
        # ---------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------------------------------

        kwargs = {"class_id": 97}
        cabinet4 = URDFObject(filename=cabinet_filename, name="cab4", category="cabinet",
                             scale=self.cabinet_scale, **kwargs)
        env.simulator.import_object(cabinet4)

        # cabinet.set_position([0,0,0.5])

        # Load microwave, set position on top of the cabinet, open it, and step 100 times
        kwargs = {"class_id": 98}
        microwave4 = URDFObject(
            filename=microwave_filename, name="mic4", category="microwave", model_path=microwave_dir,
            scale=self.microwave_scale, **kwargs
        )
        env.simulator.import_object(microwave4)
        cabinet4.set_base_link_position_orientation(
            [-np.random.uniform(50, 100), -np.random.uniform(50, 100), -np.random.uniform(50, 100)], [0, 0, 0, 1])
        microwave4.set_base_link_position_orientation(
            [-np.random.uniform(50, 100), -np.random.uniform(50, 100), -np.random.uniform(50, 100)], [0, 0, 0, 1])
        self.cabinets.append(cabinet4)
        self.microwaves.append(microwave4)
        print("Load Objects4")
        # ---------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------------------------------

        kwargs = {"class_id": 99}
        cabinet5 = URDFObject(filename=cabinet_filename, name="cab5", category="cabinet",
                             scale=self.cabinet_scale, **kwargs)
        env.simulator.import_object(cabinet5)

        # cabinet.set_position([0,0,0.5])

        # Load microwave, set position on top of the cabinet, open it, and step 100 times
        kwargs = {"class_id": 100}
        microwave5 = URDFObject(
            filename=microwave_filename, name="mic5", category="microwave", model_path=microwave_dir,
            scale=self.microwave_scale, **kwargs
        )
        env.simulator.import_object(microwave5)
        cabinet5.set_base_link_position_orientation(
            [-np.random.uniform(50, 100), -np.random.uniform(50, 100), -np.random.uniform(50, 100)], [0, 0, 0, 1])
        microwave5.set_base_link_position_orientation(
            [-np.random.uniform(50, 100), -np.random.uniform(50, 100), -np.random.uniform(50, 100)], [0, 0, 0, 1])
        self.cabinets.append(cabinet5)
        self.microwaves.append(microwave5)
        print("Load Objects5")
        # ---------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------------------------------
        kwargs = {"class_id": 101}
        cabinet6 = URDFObject(filename=cabinet_filename, name="cab6", category="cabinet",
                             scale=self.cabinet_scale, **kwargs)
        env.simulator.import_object(cabinet6)

        # cabinet.set_position([0,0,0.5])

        # Load microwave, set position on top of the cabinet, open it, and step 100 times
        kwargs = {"class_id": 102}
        microwave6 = URDFObject(
            filename=microwave_filename, name="mic6", category="microwave", model_path=microwave_dir,
            scale=self.microwave_scale, **kwargs
        )
        env.simulator.import_object(microwave6)
        cabinet6.set_base_link_position_orientation(
            [-np.random.uniform(50, 100), -np.random.uniform(50, 100), -np.random.uniform(50, 100)], [0, 0, 0, 1])
        microwave6.set_base_link_position_orientation(
            [-np.random.uniform(50, 100), -np.random.uniform(50, 100), -np.random.uniform(50, 100)], [0, 0, 0, 1])
        self.cabinets.append(cabinet6)
        self.microwaves.append(microwave6)
        """

        """
    def reset_cabin_objects(self,env,cabinet,microwave):

        state_id = p.saveState()

        for i in range(150):
            
            _, initial_pos = env.scene.get_random_point(floor=self.floor_num)
            too_close_too_others = False
            for other_cabins in self.target_pos_list:
                if l2_distance(initial_pos,other_cabins) < self.object_distance:
                    too_close_too_others = True
                    break
            if too_close_too_others:
                continue

            initial_orn = np.array([0, 0, 1.5])#np.array([0, 0, np.random.uniform(0, np.pi * 2)])
            cabinet.force_wakeup()
            reset_success = env.test_valid_position(cabinet, initial_pos, initial_orn)
            
            restoreState(state_id)
            square_area = 5
            floor_map_point = env.scene.world_to_map(initial_pos)
            enough_free_space_around_cabin = env.scene.floor_map[0][floor_map_point[1]-square_area:floor_map_point[1]+square_area,\
            floor_map_point[0]-square_area:floor_map_point[0]+square_area]
            #print("SUMM DIGGA:",(enough_free_space_around_cabin != 255).sum())
            if reset_success and (enough_free_space_around_cabin != 255).sum()==0:
                break

        

        p.removeState(state_id)

        env.set_pos_orn_with_z_offset(cabinet, initial_pos, initial_orn)
        try:
            #pass
            assert microwave.states[object_states.OnTop].set_value(cabinet, True, use_ray_casting_method=True)
        except:
            microwave.set_base_link_position_orientation([-75, -100, -100], [0, 0, 0, 1])
            return False

        self.target_pos_list.append(initial_pos)
        return True
        #assert self.microwave.states[object_states.OnTop].set_value(self.cabinet, True, use_ray_casting_method=True)

    """


    """
    def load_door_material_old(self,env):

        #load door stuff -------------------------------------------------------
        #potential classes, the specific ones are extracted and tracked below
        self.door_ids = np.arange(376, 390)

        self.door_cat_to_ind = {}#{25500: 0, 25755: 1, 26010: 2, 26775: 3}
        # 27030 , 28050, 26775, 28305, 25755, 26010, 25500, 27285, 28050, 26520
        #self.sim_class_to_sem_class = {100: 25500, 101: 0, 102: 26010, 103: 26265, 104: 26520}
        
        self.forbidden_door = 375
        dist_between_doors = env.mapping.map_settings[env.last_scene_id]['door_dist']
        self.door_dict = {}
        door_pos_list = []
        self.door_key_to_index = {}
        self.door_index_to_key = {}
        self.already_opened = {}

        #this dictionary is used in order to add permuations for the doors but keep their color-index 
        #self.permuation_indices
        ind = -1
        for key in env.scene.door_list.keys():
            ind += 1
            door_id = env.scene.door_list[key][0].get_body_ids()[0]
            #print("jooo",p.getBasePositionAndOrientation(env.scene.door_list[key].get_body_ids()[0]), "ID:",env.scene.door_list[key].get_body_ids())
            #input()
            door_pos_orient = p.getBasePositionAndOrientation(env.scene.door_list[key][0].get_body_ids()[0])
            pos_door = door_pos_orient[0]
            orient_door = door_pos_orient[1]

            door_too_close = False
            for j in door_pos_list:
                if l2_distance(j,pos_door) < dist_between_doors:
                    door_too_close = True
                    break

            if not door_too_close and ind < 4:
                #print("POS:",pos_door, " and :",orient_door)
                door_pos_list.append(pos_door)
                self.door_dict[key] = [door_id,pos_door,orient_door]
                #door_list[key][1] is simply the semantic class given in the simulator.py file.
                self.door_cat_to_ind[env.scene.door_list[key][1]] = ind

                self.door_key_to_index[key] = ind
                self.door_index_to_key[ind] = key
                
                #self.remove_collision_links.append(door_id)
                self.already_opened[key] = False
            else:
                #if theres a door to remove from door set, reduce the ind 
                ind -= 1
                env.scene.remove_object(env.scene.door_list[key][0])
                #otherwise the door cannot be moved. This is probably for buggy objects but still works for doors.
                env.scene.door_list[key][0].main_body_is_fixed = False
                env.scene.door_list[key][0].set_base_link_position_orientation([-np.random.uniform(50,100),-np.random.uniform(50,100),-np.random.uniform(50,100)],[0,0,0,1])
                #p.removeBody(door_id) -> causes error in simulator while syncing objects
    """        
    #----------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------  
import numpy as np


class greedy_baseline():
	def __init__(self):
		self.heuristic = "geodesic"
		self.strategy = "shortest" # ("shortest","random")
		self.use_exploration_policy = True
		self.use_frontier_algorithm = True
		#set true for using SGOLAM scheme
		self.use_sgolam_strategies = False
		

	def reset(self):
		pass
		

	def predict(self,env,obs):
		
		
		drivable_objects = obs['valid_actions']
		#needs to be adjusted in case of removing exploration or frontier algorithm via (True, False) flag in config.yaml
		doors = np.array(drivable_objects[1:5])
		cabinets = np.array(drivable_objects[5:8])
		cracker = np.array(drivable_objects[8:11])

		source = env.robots[0].base_link.get_position()[:2]
		if self.use_sgolam_strategies and env.current_step < 14:
			return 12
		elif self.use_sgolam_strategies and len(env.collision_links) > 0:
			return 13

		door_dists = [np.inf]*4
		if self.heuristic == "geodesic":
			for i, d_i in enumerate(doors):
				if d_i == 0:
					continue
				
				selected_color = env.mapping.original_door_colors[i]
				real_door_index = np.argwhere(env.mapping.door_colors[:,2]==selected_color[2])[0][0]
				door_name = env.task.door_index_to_key[real_door_index]
				pos = env.task.door_dict[door_name][1][:2]
				_, geodesic_dist = env.scene.get_shortest_path(0, source, pos,entire_path=False)
				door_dists[i] = geodesic_dist


			cab_dists = [np.inf]*3
			for i, d_i in enumerate(cabinets):
				if d_i == 0:
					continue
				
				pos=env.task.target_pos_list[i][:2]
				_, geodesic_dist = env.scene.get_shortest_path(0, source, pos,entire_path=False)
				cab_dists[i] = geodesic_dist

			cracker_dists = [np.inf]*3
			for i, d_i in enumerate(cracker):
				if d_i == 0:
					continue
				
				pos=env.task.target_pos_list[i+3][:2]
				_, geodesic_dist = env.scene.get_shortest_path(0, source, pos,entire_path=False)
				cracker_dists[i] = geodesic_dist


			all_dists = door_dists + cab_dists + cracker_dists
			
			if np.isinf(all_dists).sum() != 10:
				if self.strategy == "random":
					available_objects = np.argwhere(np.isinf(all_dists)==False)[0]
					
					action_rand = np.random.choice(np.arange(len(available_objects)))
					action = available_objects[action_rand] + 1 #zero is reserved for exploration plicy
				elif self.strategy == "shortest":
					action = np.argmin(all_dists) + 1 #zero is reserved for exploration plicy
				
			else:
				#no objects on map -> execute either frontier or exploration based on flags
				if self.use_exploration_policy:
					if self.use_frontier_algorithm:
						action = 0 if np.random.uniform() < 0.5 else 11
					else:
						action = 0 
				else:
					if self.use_frontier_algorithm:
						action = 11
					else:
						print("Error, one of the sub-behaviours needs to be active [exploration, frontier]")
						raise ValueError('Invalid flags')


			return action

			
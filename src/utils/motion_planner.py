import logging
from time import sleep, time

import numpy as np
import pybullet as p
from transforms3d import euler

log = logging.getLogger(__name__)


from igibson.external.pybullet_tools.utils import (
    control_joints,
    get_base_values,
    joint_controller,
    get_joint_positions,
    get_max_limits,
    get_min_limits,
    get_sample_fn,
    is_collision_free,
    joints_from_names,
    link_from_name,
    plan_base_motion_2d,
    plan_joint_motion,
    set_base_values_with_z,
    set_joint_positions,
    joint_controller_hold,
    velocity_control_joints
)
from igibson.objects.visual_marker import VisualMarker
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.utils.utils import l2_distance, quatToXYZW, restoreState, rotate_vector_2d
from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper
import cv2

class MotionPlanner(MotionPlanningWrapper):
    """
    Motion planner wrapper that supports both base and arm motion
    """

    def __init__(
        self,
        env=None,
        base_mp_algo="birrt",
        arm_mp_algo="birrt",
        optimize_iter=0,
        fine_motion_plan=True,
        full_observability_2d_planning=False,
        collision_with_pb_2d_planning=False,
        visualize_2d_planning=False,
        visualize_2d_result=False,
    ):
        """
        Get planning related parameters.
        """

        self.env = env
        
        body_ids = self.env.robots[0].get_body_ids()
        assert len(body_ids) == 1, "Only single-body robots are supported."
        self.robot_id = body_ids[0]

        # Types of 2D planning
        # full_observability_2d_planning=TRUE and collision_with_pb_2d_planning=TRUE -> We teleport the robot to locations and check for collisions
        # full_observability_2d_planning=TRUE and collision_with_pb_2d_planning=FALSE -> We use the global occupancy map from the scene
        # full_observability_2d_planning=FALSE and collision_with_pb_2d_planning=FALSE -> We use the occupancy_grid from the lidar sensor
        # full_observability_2d_planning=FALSE and collision_with_pb_2d_planning=TRUE -> [not suported yet]
        #self.full_observability_2d_planning = full_observability_2d_planning
        #self.collision_with_pb_2d_planning = collision_with_pb_2d_planning
        #assert not ((not self.full_observability_2d_planning) and self.collision_with_pb_2d_planning)

        

        self.robot = self.env.robots[0]
        
        self.arm_mp_algo = arm_mp_algo
        # If we plan in the map, we do not need to check rotations: a location is in collision (or not) independently
        # of the orientation. If we use pybullet, we may find some cases where the base orientation changes the
        # collision value for the same location between True/False
        
        self.optimize_iter = optimize_iter
        self.mode = self.env.mode

        self.animate = env.animate
        self.initial_height = self.env.initial_pos_z_offset_for_grasp
        self.fine_motion_plan = fine_motion_plan
        self.robot_type = self.robot.model_name

      

        if self.robot_type in ["Fetch"]:
            self.setup_arm_mp()

        self.arm_interaction_length = 0.2

        self.marker = None
        self.marker_direction = None

        if self.mode in ["gui_non_interactive", "gui_interactive"]:
            self.marker = VisualMarker(radius=0.04, rgba_color=[0, 0, 1, 1])
            self.marker_direction = VisualMarker(
                visual_shape=p.GEOM_CAPSULE,
                radius=0.01,
                length=0.2,
                initial_offset=[0, 0, -0.1],
                rgba_color=[0, 0, 1, 1],
            )
            self.env.simulator.import_object(self.marker)
            self.env.simulator.import_object(self.marker_direction)

        self.visualize_2d_planning = visualize_2d_planning
        self.visualize_2d_result = visualize_2d_result

        self.arm_ik_threshold = 1e-3 #the one working for the cabinets
        self.mp_obstacles = []
        self.mp_obstacles.extend(self.env.scene.get_body_ids())
        # Since the refactoring, the robot is another object in the scene
        # We need to remove it to not check twice for self collisions
        self.mp_obstacles.remove(self.robot_id)
    
    def simulator_sync(self):
        """Sync the simulator to renderer"""
        self.env.simulator.sync()

    def simulator_step(self,slow=False):
        """Step the simulator and sync the simulator to renderer"""
        #set step simulation to one since otherwise the planning doesnt work
        tmp_physics_timestep = self.env.simulator.physics_timestep_num
        if not slow:
            self.env.simulator.physics_timestep_num = 1
        self.env.simulator.step()
        if not slow:
            self.env.simulator.physics_timestep_num = tmp_physics_timestep
        self.simulator_sync()

    def get_ik_parameters(self):
        """
        Get IK parameters such as joint limits, joint damping, reset position, etc

        :return: IK parameters
        """
        max_limits, min_limits, rest_position, joint_range, joint_damping = None, None, None, None, None

        max_limits_arm = get_max_limits(self.robot_id, self.arm_joint_ids)
        max_limits = [0.5, 0.5] + [max_limits_arm[0]] + [0.5, 0.5] + list(max_limits_arm[1:]) + [0.05, 0.05]
        min_limits_arm = get_min_limits(self.robot_id, self.arm_joint_ids)
        min_limits = [-0.5, -0.5] + [min_limits_arm[0]] + [-0.5, -0.5] + list(min_limits_arm[1:]) + [0.0, 0.0]
        # increase torso_lift_joint lower limit to 0.02 to avoid self-collision
        min_limits[2] += 0.02
        current_position = get_joint_positions(self.robot_id, self.arm_joint_ids)
        rest_position = [0.0, 0.0] + [current_position[0]] + [0.0, 0.0] + list(current_position[1:]) + [0.01, 0.01]
        joint_range = list(np.array(max_limits) - np.array(min_limits))
        joint_range = [item + 1 for item in joint_range]
        joint_damping = [0.1 for _ in joint_range]

        return (max_limits, min_limits, rest_position, joint_range, joint_damping)

    def get_arm_joint_positions(self, arm_ik_goal,ee_orientation=None):
        """
        Attempt to find arm_joint_positions that satisfies arm_subgoal
        If failed, return None

        :param arm_ik_goal: [x, y, z] in the world frame
        :return: arm joint positions
        """
        log.debug("IK query for EE position {}".format(arm_ik_goal))
        ik_start = time()

        max_limits, min_limits, rest_position, joint_range, joint_damping = self.get_ik_parameters()

        n_attempt = 0
        max_attempt = 75
        sample_fn = get_sample_fn(self.robot_id, self.arm_joint_ids)
        base_pose = get_base_values(self.robot_id)
        state_id = p.saveState()
       
        while n_attempt < max_attempt:

            set_joint_positions(self.robot_id, self.arm_joint_ids, sample_fn())

            if ee_orientation is not None:
                arm_joint_positions = p.calculateInverseKinematics(
                    self.robot_id,
                    self.robot.eef_links[self.robot.default_arm].link_id,
                    targetPosition=arm_ik_goal,
                    targetOrientation=ee_orientation,
                    lowerLimits=min_limits,
                    upperLimits=max_limits,
                    jointRanges=joint_range,
                    restPoses=rest_position,
                    jointDamping=joint_damping,
                    # solver=p.IK_DLS,
                    maxNumIterations=100,
                )
            else:
                arm_joint_positions = p.calculateInverseKinematics(
                    self.robot_id,
                    self.robot.eef_links[self.robot.default_arm].link_id,
                    targetPosition=arm_ik_goal,
                    #targetOrientation=ee_orientation,
                    lowerLimits=min_limits,
                    upperLimits=max_limits,
                    jointRanges=joint_range,
                    restPoses=rest_position,
                    jointDamping=joint_damping,
                    # solver=p.IK_DLS,
                    maxNumIterations=100,
                )

            if self.robot_type == "Fetch":
                arm_joint_positions = np.array(arm_joint_positions)[self.robot_arm_indices]

            set_joint_positions(self.robot_id, self.arm_joint_ids, arm_joint_positions)

            dist = l2_distance(self.robot.get_eef_position(), arm_ik_goal)
            
            if dist > self.arm_ik_threshold: 
                n_attempt += 1
                continue

            # need to simulator_step to get the latest collision
            self.simulator_step()

            # simulator_step will slightly move the robot base and the objects
            set_base_values_with_z(self.robot_id, base_pose, z=self.initial_height)

            # TODO: have a princpled way for stashing and resetting object states

            # arm should not have any collision
            collision_free = is_collision_free(body_a=self.robot_id, link_a_list=self.arm_joint_ids)

            if not collision_free:
                n_attempt += 1
               
                continue

            # gripper should not have any self-collision
            collision_free = is_collision_free(
                body_a=self.robot_id,
                link_a_list=[self.robot.eef_links[self.robot.default_arm].link_id],
                body_b=self.robot_id,
            )
            if not collision_free:
                n_attempt += 1
               
                log.debug("Gripper in collision")
                continue

            
            restoreState(state_id)
            p.removeState(state_id)
            log.debug("IK Solver found a valid configuration")
            return arm_joint_positions

       
        restoreState(state_id)
        p.removeState(state_id)
      
        log.debug("IK Solver failed to find a configuration")
        return None

    def plan_arm_motion(self, arm_joint_positions, override_fetch_collision_links=False):
        """
        Attempt to reach arm arm_joint_positions and return arm trajectory
        If failed, reset the arm to its original pose and return None

        :param arm_joint_positions: final arm joint position to reach
        :param override_fetch_collision_links: if True, include Fetch hand and finger collisions while motion planning
        :return: arm trajectory or None if no plan can be found
        """
        log.debug("Planning path in joint space to {}".format(arm_joint_positions))
        disabled_collisions = {}
        if self.robot_type == "Fetch":
            disabled_collisions = {
                (link_from_name(self.robot_id, "torso_lift_link"), link_from_name(self.robot_id, "torso_fixed_link")),
                (link_from_name(self.robot_id, "torso_lift_link"), link_from_name(self.robot_id, "shoulder_lift_link")),
                (link_from_name(self.robot_id, "torso_lift_link"), link_from_name(self.robot_id, "upperarm_roll_link")),
                (link_from_name(self.robot_id, "torso_lift_link"), link_from_name(self.robot_id, "forearm_roll_link")),
                (link_from_name(self.robot_id, "torso_lift_link"), link_from_name(self.robot_id, "elbow_flex_link")),
            }

        if self.fine_motion_plan:
            self_collisions = True
            mp_obstacles = self.mp_obstacles
        else:
            self_collisions = False
            mp_obstacles = []

        plan_arm_start = time()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)
        state_id = p.saveState()

        allow_collision_links = []
        if self.robot_type == "Fetch" and not override_fetch_collision_links:
            allow_collision_links = [self.robot.eef_links[self.robot.default_arm].link_id] + [
                finger.link_id for finger in self.robot.finger_links[self.robot.default_arm]
            ]

        #in this function, the birrt algorothm gets (by default) executed and computed feasable
        #joint motions for satisfy the individual joint positions. For a start and end configurtion
        arm_path = plan_joint_motion(
            self.robot_id,
            self.arm_joint_ids,
            arm_joint_positions,
            disabled_collisions=disabled_collisions,
            self_collisions=self_collisions,
            obstacles=mp_obstacles,
            algorithm=self.arm_mp_algo,
            allow_collision_links=allow_collision_links,
        )
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
        restoreState(state_id)
        p.removeState(state_id)

        if arm_path is not None and len(arm_path) > 0:
            log.debug("Path found!")
        else:
            log.debug("Path NOT found!")

        return arm_path

    def dry_run_arm_plan(self, arm_path):
        """
        Dry run arm motion plan by setting the arm joint position without physics simulation

        :param arm_path: arm trajectory or None if no plan can be found
        """
        
        base_pose = get_base_values(self.robot_id)
        if arm_path is not None:
            if self.mode in ["gui_non_interactive", "gui_interactive"]:
                for joint_way_point in arm_path:
                    set_joint_positions(self.robot_id, self.arm_joint_ids, joint_way_point)
                    set_base_values_with_z(self.robot_id, base_pose, z=self.initial_height)
                    self.simulator_sync()
                    if self.animate:
                        sleep(0.02)  # animation
            else:
                set_joint_positions(self.robot_id, self.arm_joint_ids, arm_path[-1])
        else:
            set_joint_positions(self.robot_id, self.arm_joint_ids, self.arm_default_joint_positions)
        

    def plan_arm_push(self, hit_pos, hit_normal,ee_orientation=None):
        """
        Attempt to reach a 3D position and prepare for a push later

        :param hit_pos: 3D position to reach
        :param hit_normal: direction to push after reacehing that position
        :return: arm trajectory or None if no plan can be found
        """
        log.debug("Planning arm push at point {} with direction {}".format(hit_pos, hit_normal))
        

        #find configuration for joint positions which satify the goal
        # Solve the IK problem to set the arm at the desired position
        #this part just checks if they are feasable positions by using collisions etc.
        #THe actual motion gets generated below.
        joint_positions = self.get_arm_joint_positions(hit_pos,ee_orientation)
        
        
        if joint_positions is not None:
            
            # Set the arm in the default configuration to initiate arm motion planning (e.g. untucked)
            set_joint_positions(self.robot_id, self.arm_joint_ids, self.arm_default_joint_positions)
            self.simulator_sync()
            #this function will then actually generate physically feasable motions
            plan = self.plan_arm_motion(joint_positions)
            return plan
        else:
            
            log.debug("Planning failed: goal position may be non-reachable")
            return None

    def interact(self, push_point, push_direction,ee_orient=None,ee_grippers=None,doors=False):
        """
        Move the arm starting from the push_point along the push_direction
        and physically simulate the interaction

        :param push_point: 3D point to start pushing from
        :param push_direction: push direction
        """
        if doors:#the one used for door opening (hard)
            push_vector = np.array(push_direction) * 1.7#self.arm_interaction_length#*1500
            steps = 16
        else:
            push_vector = np.array(push_direction) * self.arm_interaction_length
            steps = 100

        
        max_limits, min_limits, rest_position, joint_range, joint_damping = self.get_ik_parameters()
        base_pose = get_base_values(self.robot_id)

        

        for i in range(steps):
            
            push_goal = np.array(push_point) + push_vector * (i + 1) / float(steps)

            if ee_orient is not None:
                joint_positions = p.calculateInverseKinematics(
                    self.robot_id,
                    self.robot.eef_links[self.robot.default_arm].link_id,
                    targetPosition=push_goal,
                    targetOrientation=ee_orient,
                    lowerLimits=min_limits,
                    upperLimits=max_limits,
                    jointRanges=joint_range,
                    restPoses=rest_position,
                    jointDamping=joint_damping,
                    # solver=p.IK_DLS,
                    maxNumIterations=100,
                )
            else:
                joint_positions = p.calculateInverseKinematics(
                    self.robot_id,
                    self.robot.eef_links[self.robot.default_arm].link_id,
                    targetPosition=push_goal,
                    #targetOrientation=ee_orient,
                    lowerLimits=min_limits,
                    upperLimits=max_limits,
                    jointRanges=joint_range,
                    restPoses=rest_position,
                    jointDamping=joint_damping,
                    # solver=p.IK_DLS,
                    maxNumIterations=100,
                )

            
            joint_positions = np.array(joint_positions)[self.robot_arm_indices]
           
            control_joints(self.robot_id, self.arm_joint_ids, joint_positions)
            
            if ee_grippers is not None:
                
                ee_grippers[0] -= 0.002
                ee_grippers[1] -= 0.002
                
                control_joints(self.robot_id, [20,21], ee_grippers)
            
            
            self.simulator_step(False)
            
            set_base_values_with_z(self.robot_id, base_pose, z=self.initial_height)
            
            if self.animate:# == "gui_interactive":
                self.env.robots[0]._joints['head_pan_joint'].reset_state(0.0,0.0)
                self.env.robots[0]._joints['head_tilt_joint'].reset_state(0.0,0.0)
                
                sensors = self.env.get_sensor_obs()
                ego1, ego2 = self.env.mapping.run_mapping(self.env, sensors, action=None, stop_plate_recognition=True)
                cv2.imwrite('data/vid{}/coarse/{}_{}'.format(self.env.current_episode,self.env.global_counter,'.png'),ego2.astype(np.uint8))
                cv2.imwrite('data/vid{}/fine/{}_{}'.format(self.env.current_episode,self.env.global_counter,'.png'),ego1.astype(np.uint8))
                cv2.imwrite('data/vid{}/rgb/{}_{}'.format(self.env.current_episode,self.env.global_counter,'.png'),cv2.cvtColor(sensors['rgb']*255,cv2.COLOR_RGB2BGR))
                self.env.global_counter += 1
                sleep(0.02)  # for visualization

    
    def perform_gripping(self):
        base_pose = get_base_values(self.robot_id)
        joints = self.env.robots[0]._joints.values()
        already_collided = [False,False]
        pos_indices = [0.0,0.0]
        min_collisions = 0
        num_steps = 0
        while min_collisions < 40 and num_steps < 650:
            self.env.robots[0].keep_still()
            
            num_steps += 1
            for i,j in enumerate(joints):
                if i == 12 and not already_collided[0]:
                    pos = j.get_state()[0]
                    pos_indices[0] = pos-0.002
                   
                elif i == 13 and not already_collided[1]:
                    pos = j.get_state()[0]
                    pos_indices[1] = pos-0.002
                    
            
            #velocity_control_joints(self.robot_id, [20,21], [0.5,0.5])
            control_joints(self.robot_id, [20,21], pos_indices)

            self.simulator_step(False)#self.env.simulator.step()
            set_base_values_with_z(self.robot_id, base_pose, z=self.initial_height)

           
            contact_data, robot_contact_links = self.env.robots[0]._find_gripper_contacts()
           
            
            if (35,0) in contact_data or (35,1) in contact_data:
                min_collisions += 1
                

            if (36,0) in contact_data or (36,1) in contact_data:
                
                min_collisions += 1
                
        return pos_indices  

    def execute_arm_push(self, plan, hit_pos, hit_normal,ee_orient=None,doors=False,gripper_usage=False):
        """
        Execute arm push given arm trajectory
        Should be called after plan_arm_push()

        :param plan: arm trajectory or None if no plan can be found
        :param hit_pos: 3D position to reach
        :param hit_normal: direction to push after reacehing that position
        """
        if plan is not None:
            

            log.debug("Teleporting arm along the trajectory. No physics simulation")
            self.dry_run_arm_plan(plan)
            log.debug("Performing pushing actions")
           
            pos_indices = None
            if gripper_usage:
                
                pos_indices = self.perform_gripping()
                

            self.interact(hit_pos, hit_normal,ee_orient,pos_indices,doors)
            log.debug("Teleporting arm to the default configuration")
            set_joint_positions(self.robot_id, self.arm_joint_ids, self.arm_default_joint_positions)
            self.simulator_sync()

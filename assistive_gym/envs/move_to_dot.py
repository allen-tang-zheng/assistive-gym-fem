import numpy as np
import pybullet as p
import os

from .env import AssistiveEnv

class MoveToDotEnv(AssistiveEnv):
    def __init__(self, robot, human):
        #super(MoveToDotEnv, self).__init__(robot=robot, human=human, task='scratch_itch', obs_robot_len=200*150, obs_human_len=0)
        super(MoveToDotEnv, self).__init__(robot=robot, human=human, task='scratch_itch', obs_robot_len=[270,480,3], obs_human_len=0)

    def step(self, action):
        '''agent = self.robot
        joint = agent.right_end_effector if 'right' in agent.controllable_joints else agent.left_end_effector
        ik_indices = agent.right_arm_ik_indices if 'right' in agent.controllable_joints else agent.left_arm_ik_indices
        # NOTE: Adding action to current pose can cause drift over time
        pos, orient = agent.get_pos_orient(joint)
        # NOTE: Adding action to target pose can cause large targets far outside of the robot's work space that take a long time to come back from
        # pos, orient = np.copy(agent.target_ee_position), np.copy(agent.target_ee_orientation)
        # print('Reached pos:', pos, 'Reached orient:', orient)
        # print('Reached pos:', pos, 'Reached orient:', self.get_euler(orient))
        pos += action[:len(pos)]
        orient += action[len(pos):]
        # orient = self.get_quaternion(self.get_euler(orient) + action[len(pos):len(pos)+3]) # NOTE: RPY
        # print('Target pos:', pos, 'Target orient:', orient)
        # print('Target pos:', pos, 'Target orient:', self.get_euler(orient) + action[len(pos):len(pos)+3])
        agent_joint_angles = agent.ik(joint, pos, orient, ik_indices, max_iterations=200, use_current_as_rest=True)
        #self.take_step(agent_joint_angles)'''
    
        '''self.take_step([1,1,1,1,1,1,1])
        #self.create_sphere(radius=0.01, pos=self.tool.get_pos_orient(1)[0], visual=True, collision=False, rgba=[1,1,0,1])
        #self.take_step(np.append(self.sphere_pos - self.tool.get_pos_orient(1)[0] + np.array([0,0,0.5]),self.get_quaternion([0,0,-1]) - self.tool.get_pos_orient(1)[1]), ik=True, action_multiplier=1)
        #self.take_step(np.append(np.array([-0.5,-0.3,1]) - self.tool.get_pos_orient(1)[0] + np.array([0,0,0.5]),self.get_quaternion([0,0,0]) + self.tool.get_pos_orient(1)[1] - self.tool.get_pos_orient(1)[1]), ik=True, action_multiplier=1)
        #self.take_step(np.append(np.array([0,0,-1]), self.get_quaternion([0,0,0])), ik=True, action_multiplier=1)
        print(self.tool.get_pos_orient(1)[0])
        print(np.array([0,0,1]) - self.tool.get_pos_orient(1)[0])
        #print(self.sphere_pos - self.tool.get_pos_orient(1)[0])
        print(self.tool.get_pos_orient(1)[1])
        print(self.get_quaternion([0,0,0]))'''
        
        if self.human.controllable:
            action = np.concatenate([action['robot'], action['human']])
        self.take_step(action)

        obs = self._get_obs()
        # print(np.array_str(obs, precision=3, suppress_small=True))

        tool_pos = self.tool.get_pos_orient(1)[0]
        
        total_force_on_sphere = 0
        tool_force_on_sphere = 0
        for linkA, linkB, posA, posB, force in zip(*self.tool.get_contact_points(self.sphere)):
            total_force_on_sphere += force
            if linkA in [1]:
                tool_force_on_sphere += force
        reward_distance = -np.linalg.norm(self.sphere_pos - tool_pos) # Penalize distances away from target
        reward_force = tool_force_on_sphere
        #reward_force = 0
        #if(tool_force_on_sphere > 8 and tool_force_on_sphere < 10):
        #    reward_force = 1
        print(reward_force) 
        reward = reward_distance + reward_force

        if self.gui:
            np.set_printoptions(precision=3)
            print('Task reward:', f"{reward:.03g}", 'Total force:', f"{tool_force_on_sphere:.03g}", 'Read force:', self.robot.get_force_torque_sensor(self.robot.left_end_effector-1))

        info = {'task_success': int(self.task_success >= self.config('task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = self.iteration >= 200

        if not self.human.controllable:
            return obs, reward, done, info
        else:
            # Co-optimization with both human and robot controllable
            return obs, {'robot': reward, 'human': reward}, {'robot': done, 'human': done, '__all__': done}, {'robot': info, 'human': info}

    def _get_obs(self, agent=None):
        force_torque = self.robot.get_force_torque_sensor(self.robot.left_end_effector-1)
        w, h, rgb, depth, mask = p.getCameraImage(self.camera_width, self.camera_height, self.view_matrix, self.projection_matrix, p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
        return {'visual': np.array(rgb)[:,:,0:3]/255, 'force_torque': force_torque, 'mask': mask}
        #return {"image": img/255, "force_torque": force_torque}

    def reset(self):
        super(MoveToDotEnv, self).reset()
        self.setup_camera(camera_eye=[0.5, -1, 1.5], camera_target=[-0.2, 0, 0.25], fov=60, camera_width=1920//4, camera_height=1080//4)
        plane = p.loadURDF(os.path.join(self.directory, 'plane', 'plane.urdf'), physicsClientId=self.id)
        self.plane.init(plane, self.id, self.np_random, indices=-1)
        # Randomly set friction of the ground
        self.plane.set_frictions(self.plane.base, lateral_friction=self.np_random.uniform(0.025, 0.5), spinning_friction=0, rolling_friction=0)
        # Disable rendering during creation
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.id)
        # Create robot
        self.robot.init(self.directory, self.id, self.np_random, fixed_base=not self.robot.mobile)
        self.agents.append(self.robot)
        self.robot.set_base_pos_orient([-0.5, 0.25, 0], [0, 0, -np.pi/2.0])

        # Initialize the tool in the robot's gripper
        self.tool.init(self.robot, self.task, self.directory, self.id, self.np_random, right=False, mesh_scale=[0.001]*3)

        #target_ee_pos = np.array([-0.6, 0, 0.8]) + self.np_random.uniform(-0.05, 0.05, size=3)
        #target_ee_orient = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task])
        #self.init_robot_pose(target_ee_pos, target_ee_orient, [(target_ee_pos, target_ee_orient)], [([0, 0, 0], None)], arm='left', tools=[self.tool], collision_objects=[self.furniture])

        # Open gripper to hold the tool
        self.robot.set_gripper_open_position(self.robot.left_gripper_indices, self.robot.gripper_pos[self.task], set_instantly=True)
        self.robot.enable_force_torque_sensor(self.robot.left_end_effector-1)
        
        #self.sphere_pos = [-0.593, -0.128, 0.511]
        #self.sphere_pos = [-0.51, 0.813, 0.325]
        self.sphere_pos = [-0.5,-0.3,0]
        self.sphere = self.create_sphere(radius=0.1, pos=self.sphere_pos, visual=True, collision=True, rgba=[1,0,0,1])
        #self.sphere = self.create_sphere(radius=0.04, pos=[-0.5,-0.3,0], visual=True, collision=False, rgba=[1,0,0,1])

        if not self.robot.mobile:
            self.robot.set_gravity(0, 0, 0)
        self.tool.set_gravity(0, 0, 0)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        self.init_env_variables()
        return self._get_obs()

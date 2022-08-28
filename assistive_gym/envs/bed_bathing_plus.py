import time
import numpy as np
import pybullet as p

from .env import AssistiveEnv
from .agents import furniture
from .agents.furniture import Furniture
from .agents.human_mesh import HumanMesh

class BedBathingPlusEnv(AssistiveEnv):
    def __init__(self, robot, human):
        super(BedBathingPlusEnv, self).__init__(robot=robot, human=human, task='bed_bathing', obs_robot_len=[270, 480, 3], obs_human_len=(18 + (len(human.controllable_joint_indices) if human is not None else 0)))
        self.use_mesh = (human is None)

    def step(self, action):
        if self.human.controllable:
            # action['human'][0] = 1
            # action['human'][1:] = 0
            action = np.concatenate([action['robot'], action['human']])
        self.take_step(action, ik=True)

        obs = self._get_obs()

        # Get human preferences
        end_effector_velocity = np.linalg.norm(self.robot.get_velocity(self.robot.left_end_effector))
        preferences_score = self.human_preferences(end_effector_velocity=end_effector_velocity, total_force_on_human=self.total_force_on_human, tool_force_at_target=self.tool_force_on_human)

        reward_distance = -min(self.tool.get_closest_points(self.human, distance=5.0)[-1])
        reward_action = -np.linalg.norm(action) # Penalize actions
        reward_new_contact_points = self.new_contact_points # Reward new contact points on a person

        reward = self.config('distance_weight')*reward_distance + self.config('action_weight')*reward_action + self.config('wiping_reward_weight')*reward_new_contact_points + preferences_score

        if self.gui and self.tool_force_on_human > 0:
            print('Task success:', self.task_success, 'Force at tool on human:', self.tool_force_on_human, reward_new_contact_points)

        info = {'total_force_on_human': self.total_force_on_human, 'task_success': min(self.task_success / (self.total_target_count*self.config('task_success_threshold')), 1), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = self.iteration >= 200

        if not self.human.controllable:
            return obs, reward, done, info
        else:
            # Co-optimization with both human and robot controllable
            return obs, {'robot': reward, 'human': reward}, {'robot': done, 'human': done, '__all__': done}, {'robot': info, 'human': info}

    def get_total_force(self):
        total_force_on_human = np.sum(self.robot.get_contact_points(self.human)[-1])
        tool_force = np.sum(self.tool.get_contact_points()[-1])
        tool_force_on_human = 0
        new_contact_points = 0
        for linkA, linkB, posA, posB, force in zip(*self.tool.get_contact_points(self.human)):
            total_force_on_human += force
            if linkA in [1]:
                tool_force_on_human += force
                # Only consider contact with human upperarm, forearm, hand
                if linkB < 0 or linkB > len(self.human.all_joint_indices):
                    continue

                indices_to_delete = []
                for i, (target_pos_world, target) in enumerate(zip(self.targets_pos_upperarm_world, self.targets_upperarm)):
                    if np.linalg.norm(posB - target_pos_world) < 0.025:
                        # The robot made contact with a point on the person's arm
                        new_contact_points += 1
                        self.task_success += 1
                        target.set_base_pos_orient(self.np_random.uniform(1000, 2000, size=3), [0, 0, 0, 1])
                        indices_to_delete.append(i)
                self.targets_pos_on_upperarm = [t for i, t in enumerate(self.targets_pos_on_upperarm) if i not in indices_to_delete]
                self.targets_upperarm = [t for i, t in enumerate(self.targets_upperarm) if i not in indices_to_delete]
                self.targets_pos_upperarm_world = [t for i, t in enumerate(self.targets_pos_upperarm_world) if i not in indices_to_delete]

                indices_to_delete = []
                for i, (target_pos_world, target) in enumerate(zip(self.targets_pos_forearm_world, self.targets_forearm)):
                    if np.linalg.norm(posB - target_pos_world) < 0.025:
                        # The robot made contact with a point on the person's arm
                        new_contact_points += 1
                        self.task_success += 1
                        target.set_base_pos_orient(self.np_random.uniform(1000, 2000, size=3), [0, 0, 0, 1])
                        indices_to_delete.append(i)
                self.targets_pos_on_forearm = [t for i, t in enumerate(self.targets_pos_on_forearm) if i not in indices_to_delete]
                self.targets_forearm = [t for i, t in enumerate(self.targets_forearm) if i not in indices_to_delete]
                self.targets_pos_forearm_world = [t for i, t in enumerate(self.targets_pos_forearm_world) if i not in indices_to_delete]

        return tool_force, tool_force_on_human, total_force_on_human, new_contact_points

    def _get_obs(self, agent=None):
        self.tool_force, self.tool_force_on_human, self.total_force_on_human, self.new_contact_points = self.get_total_force()
        force_torque = self.robot.get_force_torque_sensor(self.robot.left_end_effector-1)
        w, h, rgb, depth, mask = p.getCameraImage(self.camera_width, self.camera_height, self.view_matrix, self.projection_matrix, p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
        return {'visual': np.array(rgb)[:,:,0:3]/255, 'force_torque': force_torque, 'mask': mask}

    def reset(self):
        super(BedBathingPlusEnv, self).reset()
        self.setup_camera(camera_eye=[0.5, -0.75, 1.5], camera_target=[-0.2, 0, 0.75], fov=60, camera_width=1920//4, camera_height=1080//4)
        self.build_assistive_env('hospital_bed', fixed_human_base=False)
        self.furniture.set_on_ground()

        self.furniture.set_joint_angles([1], [np.pi/4])
        self.furniture.set_whole_body_frictions(1, 1, 1)

        # Setup human in the air and let them settle into a resting pose on the bed
        if self.use_mesh:
            self.human = HumanMesh()
            joints_positions = []
            #joints_positions = [(self.human.j_right_shoulder_z, 50), (self.human.j_right_elbow_z, 30), (self.human.j_left_shoulder_x, -30), (self.human.j_waist_x, 45)]
            body_shape = np.zeros((1, 10))
            gender = 'female' # 'random'
            self.human.init(self.directory, self.id, self.np_random, gender=gender, height=None, body_shape=body_shape, joint_angles=joints_positions, left_hand_pose=[[-2, 0, 0, -2, 0, 0]])

            #self.human.set_base_pos_orient([0, -0.05, 0.85], [-np.pi/2.0, 0, 0])
            self.human.set_base_pos_orient([0, -0.05, 0.85], [0, 0, 0])
        else:
            joints_positions = [(self.human.j_right_shoulder_x, 30), (self.human.j_left_shoulder_x, -30), (self.human.j_waist_x, 45)]
            self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=None)
            self.human.set_base_pos_orient([0, -0.05, 0.85], [-np.pi/2.0, 0, 0])
        
        # Stiffen human chest joints to prevent the robot from collapsing in on itself when hitting the bed
        # stiffness = self.human.get_joint_stiffness()
        # for joint in [self.human.j_waist_x, self.human.j_waist_y, self.human.j_waist_z, self.human.j_chest_x, self.human.j_chest_y, self.human.j_chest_z, self.human.j_upper_chest_x, self.human.j_upper_chest_y, self.human.j_upper_chest_z]:
        #     self.human.set_joint_stiffness(joint, 1)

        p.setGravity(0, 0, -1, physicsClientId=self.id)

        # Add small variation in human joint positions
        #when using smplx model it yells about something i don't understand so now its commented out
        if(not self.use_mesh):
            motor_indices, motor_positions, motor_velocities, motor_torques = self.human.get_motor_joint_states()
            self.human.set_joint_angles(motor_indices, motor_positions+self.np_random.uniform(-0.1, 0.1, size=len(motor_indices)))

        # Let the person settle on the bed
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)
            # time.sleep(0.5)

        # Reset joint stiffness to default values
        # for joint, s in zip(self.human.all_joint_indices, stiffness):
        #     self.human.set_joint_stiffness(joint, s)

        # Stiffen uncontrolled joints and freeze the human base on the bed
        if(not self.use_mesh):
            self.human.control(self.human.all_joint_indices, self.human.get_joint_angles(), 0.05, 100) #still yelling about joint_states
        self.human.set_mass(self.human.base, mass=0)
        self.human.set_base_velocity(linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])

        if(self.use_mesh):
            shoulder_pos = self.human.get_pos_orient(self.human.right_shoulder)[0]
            elbow_pos = self.human.get_pos_orient(self.human.right_elbow)[0]
            wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]
        else:
            shoulder_pos = self.human.get_pos_orient(self.human.right_upperarm)[0]
            elbow_pos = self.human.get_pos_orient(self.human.right_forearm)[0]
            wrist_pos = self.human.get_pos_orient(self.human.right_hand)[0]

        # Initialize the tool in the robot's gripper
        self.tool.init(self.robot, self.task, self.directory, self.id, self.np_random, right=False, mesh_scale=[1]*3)
        target_ee_pos = np.array([-0.45, -0.1, 1]) + self.np_random.uniform(-0.05, 0.05, size=3)
        target_ee_orient = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task])
        base_position = self.init_robot_pose(target_ee_pos, target_ee_orient, [(target_ee_pos, target_ee_orient)], [(shoulder_pos, None), (elbow_pos, None), (wrist_pos, None)], arm='left', tools=[self.tool], collision_objects=[self.human, self.furniture], wheelchair_enabled=False)

        if self.robot.wheelchair_mounted:
            # Load a nightstand in the environment for mounted arms
            self.nightstand = Furniture()
            self.nightstand.init('nightstand', self.directory, self.id, self.np_random)
            self.nightstand.set_base_pos_orient(np.array([-0.9, 0.4, 0]) + base_position, [0, 0, 0, 1])

        # Open gripper to hold the tool
        self.robot.set_gripper_open_position(self.robot.left_gripper_indices, self.robot.gripper_pos[self.task], set_instantly=True)

        self.robot.enable_force_torque_sensor(self.robot.left_end_effector-1)
        self.generate_targets()

        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        if not self.robot.mobile:
            self.robot.set_gravity(0, 0, 0)
        self.human.set_gravity(0, 0, 0)
        self.tool.set_gravity(0, 0, 0)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        self.init_env_variables()
        return self._get_obs()

    def generate_targets(self):
        self.target_indices_to_ignore = []
        if(self.use_mesh):
            self.upperarm, self.upperarm_length, self.upperarm_radius = self.human.right_shoulder, np.linalg.norm(self.human.get_pos_orient(self.human.right_shoulder)[0] - self.human.get_pos_orient(self.human.right_elbow)[0]), self.human.shoulder_radius
        else:
            self.upperarm, self.upperarm_length, self.upperarm_radius = self.human.right_upperarm, self.human.upperarm_length, self.human.upperarm_radius
        if(self.use_mesh):
            self.forearm, self.forearm_length, self.forearm_radius = self.human.right_elbow, np.linalg.norm(self.human.get_pos_orient(self.human.right_elbow)[0] - self.human.get_pos_orient(self.human.right_wrist)[0]), self.human.elbow_radius
        else:
            self.forearm, self.forearm_length, self.forearm_radius = self.human.right_forearm, self.human.forearm_length, self.human.forearm_radius
        if(self.use_mesh):
            self.targets_pos_on_upperarm = self.util.capsule_points(p1=np.array([0, 0, 0.06411867825041404]), p2=np.array([0, 0, -self.upperarm_length]), radius=self.upperarm_radius, distance_between_points=0.03, num_sections=7, points_per_section=10)
        else:
            self.targets_pos_on_upperarm = self.util.capsule_points(p1=np.array([0, 0, -self.human.pecs_offset]), p2=np.array([0, 0, -self.upperarm_length]), radius=self.upperarm_radius, distance_between_points=0.03, num_sections=7, points_per_section=10)
        self.targets_pos_on_forearm = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.forearm_length]), radius=self.forearm_radius, distance_between_points=0.03, num_sections=7, points_per_section=6)

        self.targets_upperarm = self.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.targets_pos_on_upperarm), visual=True, collision=False, rgba=[0, 1, 1, 1])
        self.targets_forearm = self.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.targets_pos_on_forearm), visual=True, collision=False, rgba=[0, 1, 1, 1])
        self.total_target_count = len(self.targets_pos_on_upperarm) + len(self.targets_pos_on_forearm)
        self.update_targets()

    def update_targets(self):
        upperarm_pos, upperarm_orient = self.human.get_pos_orient(self.upperarm)
        self.targets_pos_upperarm_world = []
        for target_pos_on_arm, target in zip(self.targets_pos_on_upperarm, self.targets_upperarm):
            #print(p.getQuaternionFromEuler(self.human.get_pos_orient(self.human.right_shoulder)[0] - self.human.get_pos_orient(self.human.right_elbow)[0]))
            #print(self.human.get_pos_orient(self.human.right_shoulder)[0] - self.human.get_pos_orient(self.human.right_elbow)[0])
            target_pos = np.array(p.multiplyTransforms(upperarm_pos, upperarm_orient, target_pos_on_arm, [0, 0, 0, 1], physicsClientId=self.id)[0])
            self.targets_pos_upperarm_world.append(target_pos)
            target.set_base_pos_orient(target_pos, [0, 0, 0, 1])

        forearm_pos, forearm_orient = self.human.get_pos_orient(self.forearm)
        self.targets_pos_forearm_world = []
        for target_pos_on_arm, target in zip(self.targets_pos_on_forearm, self.targets_forearm):
            target_pos = np.array(p.multiplyTransforms(forearm_pos, forearm_orient, target_pos_on_arm, [0, 0, 0, 1], physicsClientId=self.id)[0])
            self.targets_pos_forearm_world.append(target_pos)
            target.set_base_pos_orient(target_pos, [0, 0, 0, 1])


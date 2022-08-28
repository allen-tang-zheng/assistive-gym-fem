from .bed_bathing import BedBathingEnv
from .agents import pr2, baxter, sawyer, jaco, stretch, panda, human
from .agents.pr2 import PR2
from .agents.baxter import Baxter
from .agents.sawyer import Sawyer
from .agents.jaco import Jaco
from .agents.stretch import Stretch
from .agents.panda import Panda
from .agents.human import Human
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from .bed_bathing_plus import BedBathingPlusEnv
from .move_to_dot import MoveToDotEnv

robot_arm = 'left'
human_controllable_joint_indices = human.body_joints + human.right_arm_joints
class MoveToDotJacoEnv(MoveToDotEnv):
    def __init__(self):
        super(MoveToDotJacoEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))
        #super(BedBathingPlusJacoEnv, self).__init__(robot=Jaco(robot_arm), human=None)

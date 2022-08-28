from .dressing import DressingEnv
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
from .dressing_plus import DressingPlusEnv

robot_arm = 'left'
human_controllable_joint_indices = human.left_arm_joints
class DressingPlusPR2MeshEnv(DressingPlusEnv):
    def __init__(self):
        super(DressingPlusPR2MeshEnv, self).__init__(robot=PR2(robot_arm), human=None)


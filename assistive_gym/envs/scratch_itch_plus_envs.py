from .scratch_itch import ScratchItchEnv
from .scratch_itch_mesh import ScratchItchMeshEnv
from .agents import pr2, baxter, sawyer, jaco, stretch, panda, human, human_mesh
from .agents.pr2 import PR2
from .agents.baxter import Baxter
from .agents.sawyer import Sawyer
from .agents.jaco import Jaco
from .agents.stretch import Stretch
from .agents.panda import Panda
from .agents.human import Human
from .agents.human_mesh import HumanMesh
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from .scratch_itch_plus import ScratchItchPlusEnv

robot_arm = 'left'
human_controllable_joint_indices = human.right_arm_joints
class ScratchItchPlusJacoEnv(ScratchItchPlusEnv):
    def __init__(self):
        super(ScratchItchPlusJacoEnv, self).__init__(robot=Jaco(robot_arm), human=None)
        #super(ScratchItchPlusJacoEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))


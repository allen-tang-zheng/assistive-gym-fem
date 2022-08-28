import ray
import ray.rllib.agents.ppo as ppo
from ray import tune
import torch.nn as nn
import torch
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

def train_ppo_model():
    config = ppo.DEFAULT_CONFIG.copy()
    config["framework"] = "torch"
    trainer = ppo.PPOTrainer(
        config=config,
        env="CartPole-v0",
    )
    
    checkpoint = torch.load("./rllib_checkpoint/model.pt")
    trainer.workers.local_worker().get_policy().model.load_state_dict(checkpoint)
    trainer.workers.sync_weights()
    print("============")
    print(trainer.get_policy().model.state_dict()['_value_branch._model.0.weight'][0,0:5])
    
    # Train for one iteration
    result = trainer.train()
    print(f"Iteration: {result['training_iteration']}, total timesteps: {result['timesteps_total']}, total time: {result['time_total_s']:.1f}, FPS: {result['timesteps_total']/result['time_total_s']:.1f}, mean reward: {result['episode_reward_mean']:.1f}, min/max reward: {result['episode_reward_min']:.1f}/{result['episode_reward_max']:.1f}")
    path = None
    #path = trainer.save("./rllib_checkpoint/")
    print("------------")
    print(trainer.get_policy().model.state_dict()['_value_branch._model.0.weight'][0,0:5])
    torch.save(trainer.get_policy().model.state_dict(), "./rllib_checkpoint/model.pt")
    # trainer.restore("./rllib_checkpoint/checkpoint_000001/checkpoint-1")
    return path


ray.shutdown()
ray.init(ignore_reinit_error=True)

checkpoint_path = train_ppo_model()
reward = [1, 2, 3]
rewarda = [2,3,4]
rewardb = [0,1,2]
fig = plt.figure()
ax = fig.add_subplot(111)
fig.show()
ax.clear
ax.plot(reward)
ax.plot(rewarda)
ax.plot(rewardb)
fig.canvas.draw()
plt.show()

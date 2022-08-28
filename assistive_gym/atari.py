import os, sys, multiprocessing, gym, ray, shutil, argparse, importlib, glob
import torch
import torch.nn as nn
import numpy as np
# from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.agents import ppo, sac
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.logger import pretty_print
from numpngw import write_apng

ray.shutdown()
ray.init(ignore_reinit_error=True)

num_processes = multiprocessing.cpu_count()

config = {"env": "StarGunnerNoFrameskip-v4", "num_workers": 10,"framework": "torch","num_gpus": 1, "num_envs_per_worker": 5, "lambda" : 0.95, "kl_coeff" : 0.5, "clip_param" : 0.1, "vf_clip_param" : 200.0, "entropy_coeff" : 0.01, "train_batch_size" : 5000, "rollout_fragment_length": 100, "sgd_minibatch_size": 500, "num_sgd_iter" : 8, "clip_rewards": True, "batch_mode":"truncate_episodes","observation_filter":"NoFilter"}
agent = ppo.PPOTrainer(config)
#ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
#env = gym.make('CartPole-v1')
#env.disconnect()

timesteps = 0
while timesteps < 10000:
    result = agent.train()
    timesteps = result['timesteps_total']
    torch.cuda.memory_summary(device=None, abbreviated=False)
    print(f"Iteration: {result['training_iteration']}, total timesteps: {result['timesteps_total']}, total time: {result['time_total_s']:.1f}, FPS: {result['timesteps_total']/result['time_total_s']:.1f}, mean reward: {result['episode_reward_mean']:.1f}, min/max reward: {result['episode_reward_min']:.1f}/{result['episode_reward_max']:.1f}")
    sys.stdout.flush()

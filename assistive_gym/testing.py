import os, sys, multiprocessing, gym, ray, shutil, argparse, importlib, glob
import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
# from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.agents import ppo, sac
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.logger import pretty_print
from numpngw import write_apng
        
class SimpleConvAgent(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self._num_objects = 2
        self._num_actions = num_outputs

        linear_flatten = np.prod(obs_space.shape[:2])*64

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(self._num_objects, 32, 3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(linear_flatten, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 512)),
            nn.ReLU(),
        )

        self._actor_head = nn.Sequential(
            layer_init(nn.Linear(512, 256), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(256, self._num_actions), std=0.01)
        )

        self._critic_head = nn.Sequential(
            layer_init(nn.Linear(512, 1), std=0.01)
        )

    def forward(self, input_dict, state, seq_lens):
        obs_transformed = input_dict['obs'].permute(0, 3, 1, 2)
        network_output = self.network(obs_transformed)
        value = self._critic_head(network_output)
        self._value = value.reshape(-1)
        logits = self._actor_head(network_output)
        return logits, state

    def value_function(self):
        return self._value

class ConvVisual(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.visual_model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_outputs)
        )
        
        self.value = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )
         
    def forward(self, input_dict, state, seq_lens):
        self.vis = torch.permute(input_dict["obs"]["visual"].float(), (0,3,1,2))
        self.force = input_dict["obs"]["force_torque"].float()
        self.res = self.visual_model(self.vis)
        return self.res, state
        #return self.process_image.forward(obs)
        
    def value_function(self):
        self.res = self.value(self.vis)
        return self.res.squeeze(1)

class ConvAppend(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.visual_model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.ReLU(inplace=True),
        )
        
        self.tail = nn.Sequential(
            nn.Linear(128+6, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_outputs)
        )
        
        self.value = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.ReLU(inplace=True),
        )
        
        self.value_tail = nn.Sequential(
            nn.Linear(128+6, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )
         
    def forward(self, input_dict, state, seq_lens):
        self.vis = torch.permute(input_dict["obs"]["visual"].float(), (0,3,1,2))
        self.force = input_dict["obs"]["force_torque"].float()
        self.emb = self.visual_model(self.vis)
        self.cat = torch.cat((self.emb, self.force), -1)
        self.res = self.tail(self.cat)
        return self.res, state
        
    def value_function(self):
        self.emb_value = self.value(self.vis)
        self.cat_value = torch.cat((self.emb_value, self.force), -1)
        return self.value_tail(self.cat_value).squeeze(1)
        
class ConvChannel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.visual_model = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=32, kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_outputs)
        )
        
        self.value = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=32, kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )
         
    def forward(self, input_dict, state, seq_lens):
        self.vis = torch.permute(input_dict["obs"]["visual"].float(), (0,3,1,2))
        self.force = input_dict["obs"]["force_torque"].float()
        self.force = self.force.view(self.force.size()[0], -1, 1, 1)
        self.channel = torch.cat((self.vis, self.force.expand(self.vis.size()[0], 6, 270, 480)), 1)
        self.res = self.visual_model(self.channel)
        return self.res, state
        
    def value_function(self):
        self.res = self.value(self.channel)
        return self.res.squeeze(1)
      
class ConvChannelRestricted(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        if(model_config["custom_model_config"]["env"] == "MoveToDot"):
            self.maks = 2
        if(model_config["custom_model_config"]["env"] == "BedBathing"):
            self.maks = 4
        if(model_config["custom_model_config"]["env"] == "ScratchItch"):
            self.maks = 4
        if(model_config["custom_model_config"]["env"] == "Dressing"):
            self.maks = 4
        
        self.visual_model = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=32, kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_outputs)
        )
        
        self.value = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=32, kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )
         
    def forward(self, input_dict, state, seq_lens):
        self.vis = torch.permute(input_dict["obs"]["visual"].float(), (0,3,1,2))
        self.force = input_dict["obs"]["force_torque"].float()
        self.force = self.force.view(self.force.size()[0], -1, 1, 1)
        self.channel = torch.cat((self.vis, self.force.expand(self.vis.size()[0], 6, 270, 480)), 1)
        self.res = self.visual_model(self.channel)
        return self.res, state
        
    def value_function(self):
        self.res = self.value(self.channel)
        return self.res.squeeze(1)
        
def setup_config():
    num_processes = 10#multiprocessing.cpu_count()
    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    config['train_batch_size'] = 2000
    config['num_sgd_iter'] = 50
    config['sgd_minibatch_size'] = 16
    config['lambda'] = 0.95
    config["num_gpus"] = 1
    config["framework"] = "torch"
    config['num_workers'] = num_processes
    config['num_cpus_per_worker'] = 0
    config['log_level'] = 'ERROR'
    #config['model']['fcnet_hiddens'] = [100, 100]
    config['model'] = {
        "custom_model": "Model",
        "custom_model_config": {"env": "MoveToDot"},
        }
    config['env'] = 'assistive_gym:MoveToDot-v0'
    return {**config}

def load_policy(mode, policy_path=None):
    agent = ppo.PPOTrainer(setup_config(), 'assistive_gym:MoveToDot-v0')
    files = [f.split('_')[-1] for f in glob.glob("./trained_models/ppo/"+mode+"/model_*.pt")]
    files_ints = [int(f[:-3]) for f in files]
    if files:
        checkpoint_max = max(files_ints)
        checkpoint = torch.load("./trained_models/ppo/"+mode+"/model_"+str(checkpoint_max)+".pt")
        agent.workers.local_worker().get_policy().model.load_state_dict(checkpoint)
        agent.workers.sync_weights()
    return agent

def render_policy(mode, n_episodes=1):
    env = gym.make('assistive_gym:MoveToDot-v0')
    test_agent = load_policy(mode)

    env.render()
    frames = []
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        tot_reward = 0
        rewards = []
        while not done:
            # Compute the next action using the trained policy
            action = test_agent.compute_action(obs)
            # Step the simulation forward using the action from our trained policy
            obs, reward, done, info = env.step(action)
            #rewards.append(reward)
            tot_reward += reward
        print("Total Reward:", tot_reward)
        #plt.plot(rewards)
        #plt.show()
    #env.disconnect()
    
def train(mode, episodes):
    agent = load_policy(mode)
    timesteps = 0
    reward_mean = []
    reward_min = []
    reward_max = []
    if os.path.exists("./trained_models/ppo/"+mode+"/reward_mean.pkl"):
        with open("./trained_models/ppo/"+mode+"/reward_mean.pkl", "rb") as fp:
            reward_mean = pickle.load(fp)
    if os.path.exists("./trained_models/ppo/"+mode+"/reward_min.pkl"):
        with open("./trained_models/ppo/"+mode+"/reward_min.pkl", "rb") as fp:
            reward_min = pickle.load(fp)
    if os.path.exists("./trained_models/ppo/"+mode+"/reward_max.pkl"):
        with open("./trained_models/ppo/"+mode+"/reward_max.pkl", "rb") as fp:
            reward_max = pickle.load(fp) 
    while timesteps < episodes:
        result = agent.train()
        timesteps = result['timesteps_total']
        reward_mean.append(result['episode_reward_mean'])
        reward_min.append(result['episode_reward_min'])
        reward_max.append(result['episode_reward_max'])
        #if checkpoint_path is not None:
        #    shutil.rmtree(os.path.dirname(checkpoint_path), ignore_errors=True)
        #checkpoint_path = agent.save(os.path.join(save_dir, 'ppo', 'MoveToDot-v0'))
        print(f"Iteration: {result['training_iteration']}, total timesteps: {result['timesteps_total']}, total time: {result['time_total_s']:.1f}, FPS: {result['timesteps_total']/result['time_total_s']:.1f}, mean reward: {result['episode_reward_mean']:.1f}, min/max reward: {result['episode_reward_min']:.1f}/{result['episode_reward_max']:.1f}")
        sys.stdout.flush()
        if(timesteps%10000 == 0):
            torch.save(agent.get_policy().model.state_dict(), "./trained_models/ppo/"+mode+"/model_"+str(timesteps)+".pt")
            print("./trained_models/ppo/"+mode+"/model_"+str(timesteps)+".pt")
            with open("./trained_models/ppo/"+mode+"/reward_mean.pkl", "wb") as fp:
                pickle.dump(reward_mean, fp)
            with open("./trained_models/ppo/"+mode+"/reward_min.pkl", "wb") as fp:
                pickle.dump(reward_min, fp)
            with open("./trained_models/ppo/"+mode+"/reward_max.pkl", "wb") as fp:
                pickle.dump(reward_max, fp)
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #fig.show()
    #ax.clear
    #ax.plot(reward_mean)
    #ax.plot(reward_min)
    #ax.plot(reward_max)
    #fig.canvas.draw()
    #plt.show()

ray.shutdown()
ray.init(ignore_reinit_error=True)

#ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
#env = gym.make('CartPole-v1')
#env.disconnect()

save_dir='./trained_models/'
render = False
graph = False 
if(render):
    ModelCatalog.register_custom_model("Model", ConvAppend)
    render_policy('visual1', 10)
elif(graph):
    a = []
    b = []
    c = []
    with open("./trained_models/ppopastbetween8and10/visual1/reward_mean.pkl", "rb") as fp:
        a = pickle.load(fp)
    with open("./trained_models/ppopastbetween8and10/visual2/reward_mean.pkl", "rb") as fp:
        b = pickle.load(fp)
    with open("./trained_models/ppopastbetween8and10/visual3/reward_mean.pkl", "rb") as fp:
        c = pickle.load(fp)
    visual = np.concatenate((np.asarray(a).reshape(-1,1), np.asarray(b).reshape(-1,1), np.asarray(c).reshape(-1,1)), axis=1)
    visual = np.sum(visual, axis=1)/3
    plt.clf()
    plt.plot(a, color='lightgreen')
    plt.plot(b, color='lightgreen')
    plt.plot(c, color='lightgreen')
    plt.plot(visual, label='visual average', linewidth=5, color='green')
    '''
    d = []
    e = []
    f = []
    with open("./trained_models/ppo/channelM1/reward_mean.pkl", "rb") as fp:
        d = pickle.load(fp)
    with open("./trained_models/ppo/channelM2/reward_mean.pkl", "rb") as fp:
        e = pickle.load(fp)
    with open("./trained_models/ppo/channelM3/reward_mean.pkl", "rb") as fp:
        f = pickle.load(fp)
    append = np.concatenate((np.asarray(d).reshape(-1,1), np.asarray(e).reshape(-1,1), np.asarray(f).reshape(-1,1)), axis=1)
    append = np.sum(append, axis=1)/3
    plt.plot(d, color='lightblue')
    plt.plot(e, color='lightblue')
    plt.plot(f, color='lightblue')
    plt.plot(append, label='channel with no force average', linewidth=5, color='blue')
    
    g = []
    h = []
    i = []
    with open("./trained_models/ppo/channel1/reward_mean.pkl", "rb") as fp:
        g = pickle.load(fp)
    with open("./trained_models/ppo/channel2/reward_mean.pkl", "rb") as fp:
        h = pickle.load(fp)
    with open("./trained_models/ppo/channel3/reward_mean.pkl", "rb") as fp:
        i = pickle.load(fp)
    append = np.concatenate((np.asarray(g).reshape(-1,1), np.asarray(h).reshape(-1,1), np.asarray(i).reshape(-1,1)), axis=1)
    append = np.sum(append, axis=1)/3
    plt.plot(g, color='moccasin')
    plt.plot(h, color='moccasin')
    plt.plot(i, color='moccasin')
    plt.plot(append, label='normal channel average', linewidth=5, color='orange')'''
    
    j = []
    k = []
    l = []
    with open("./trained_models/ppopastbetween8and10/append1/reward_mean.pkl", "rb") as fp:
        j = pickle.load(fp)
    with open("./trained_models/ppopastbetween8and10/append2/reward_mean.pkl", "rb") as fp:
        k = pickle.load(fp)
    with open("./trained_models/ppopastbetween8and10/append3/reward_mean.pkl", "rb") as fp:
        l = pickle.load(fp)
    append = np.concatenate((np.asarray(j).reshape(-1,1), np.asarray(k).reshape(-1,1), np.asarray(l).reshape(-1,1)), axis=1)
    append = np.sum(append, axis=1)/3
    plt.plot(j, color='lightyellow')
    plt.plot(k, color='lightyellow')
    plt.plot(l, color='lightyellow')
    plt.plot(append, label='append average', linewidth=5, color='gold')
    '''
    j = []
    k = []
    l = []  
    with open("./trained_models/ppo/channel1/reward_mean.pkl", "rb") as fp:
        j = pickle.load(fp)
    with open("./trained_models/ppo/channel1/reward_max.pkl", "rb") as fp:
        k = pickle.load(fp)
    with open("./trained_models/ppo/channel1/reward_min.pkl", "rb") as fp:
        l = pickle.load(fp)
    plt.plot(j)
    plt.plot(k)
    plt.plot(l)'''
    
    plt.legend()
    plt.title('training iterations vs reward (force between 8 and 10)')
    plt.show()
else:
    ModelCatalog.register_custom_model("Model", ConvVisual)
    train('visual1', 200000)
    #train('channel2', 200000)
    #train('channel3', 200000)

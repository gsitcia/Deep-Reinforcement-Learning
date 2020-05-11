import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
import wrappers

import time
import sys
import numpy as np

import random

from networks import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Network File
path = "checkpoints/%s.pt"%sys.argv[1]

# Environment
env = gym.make("Breakout-v0")
env = wrappers.make_env(env)

# Network
q_network = CNN(4,env.action_space.n).to(device)
q_network.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
q_network.eval()


# Greedy Policy
@torch.no_grad()
def select_action(q, state):
    if random.random() < 0.05:
        print('Random')
        return env.action_space.sample()
    k = q(state.to(device))
    print(k)
    return k.max(1)[1].view(1, 1)


while True:  # every episode
	obs = env.reset()
	state = torch.cat(obs).unsqueeze(0)
	while True:  # every timestep
		env.render()

		# Select Action
		action = select_action(q_network, state)
		
		# Environment Step
		obs, reward, done, info = env.step(action)
		state = torch.cat(obs).unsqueeze(0)
		time.sleep(0.05)
		
        
		if done:
			break

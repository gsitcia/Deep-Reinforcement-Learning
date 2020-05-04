import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
import wrappers

from networks import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Network File
path = "checkpoints/ep_100.pt"

# Network
q_network = DQN().to(device)
q_network.load_state_dict(torch.load(path))
q_network.eval()


# Greedy Policy
def select_action(q, state):
    with torch.no_grad():
        return q(state.to(device)).max(1)[1].view(1, 1)


# Environment
env = gym.make("Breakout-v0")
env = wrappers.make_env(env)

while True:  # every episode
    obs = env.reset()
    state = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0)
    while True:  # every timestep
        env.render()
        time.sleep(0.065)

        # Select Action
        action = select_action(q_network, state)

        # Environment Step
        obs, reward, done, info = env.step(action)

        if done:
            break

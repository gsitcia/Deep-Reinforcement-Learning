import itertools
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym

import wrappers
from networks import DQN
from replay import ReplayMemory, Transition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hparams = argparse.Namespace(n_episodes=1000,
                             eps_start=1,
                             eps_end=0.02,
                             eps_decay=1e-6,
                             target_update=1000,
                             lr=1e-4,
                             train_start=10000,
                             memory_size=1e6)


def select_action(state, global_step):
    sample = random.random()
    eps_start = hparams.eps_start
    eps_end = hparams.eps_end
    eps_decay = hparams.eps_decay

    eps = eps_end + (eps_start - hparams.ep)


# Networks
q_network = DQN().to(device)
target_network = DQN().to(device)
target_network.load_state_dict(q_network.state_dict())
target_network.eval()

# Optimizers
optimizer = torch.optim.Adam(q_network, lr=hparams.lr)

# Environment
env = gym.make("Breakout-v0")
env = wrappers.make_env(env)

global_step = 0
for episode in range(hparams.n_episodes):
    obs = env.reset()
    state = obs.permute(2, 0, 1).unsqueeze(0)
    total_reward = 0.0
    for t in itertools.count():
        global_step += 1


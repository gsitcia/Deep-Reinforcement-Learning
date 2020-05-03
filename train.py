import itertools
import argparse

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym

import wrappers
from networks import DQN
from replay import ReplayMemory, Transition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hparams = argparse.Namespace(batch_size=32,
                             n_episodes=1000,
                             gamma=0.99,
                             eps_start=1,
                             eps_end=0.02,
                             eps_decay=1e-6,
                             target_update=5000,
                             lr=2.5e-4,
                             momentum=0.95,
                             train_start=10000,
                             memory_size=1000000,
                             render=True)

assert hparams.train_start >= hparams.batch_size

def select_action(q, state, global_step):
    eps_start = hparams.eps_start
    eps_end = hparams.eps_end
    eps_decay = hparams.eps_decay

    sample = random.random()
    epsilon = eps_end + (eps_start - eps_end) * math.exp(-eps_decay * global_step)
    if sample > epsilon:
        with torch.no_grad():
            return q(state.to(device)).max(1)[1].view(1, 1)
    else:
        return torch.randint(0, 4, (1, 1))


# Networks
q_network = DQN().to(device)
target_network = DQN().to(device)
target_network.load_state_dict(q_network.state_dict())
target_network.eval()

# Replay Memory
memory = ReplayMemory(hparams.memory_size)

# Optimizers
optimizer = torch.optim.RMSprop(q_network.parameters(), lr=hparams.lr, momentum=hparams.momentum)

# Environment
env = gym.make("Breakout-v0")
env = wrappers.make_env(env)

global_step = 0
for episode in range(hparams.n_episodes):
    obs = env.reset()
    state = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0)
    total_reward = 0.0
    for t in itertools.count():
        if hparams.render:
            env.render()

        action = select_action(q_network, state, global_step)

        obs, reward, done, info = env.step(action)

        next_state = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0) if not done else None

        total_reward += reward
        reward = torch.tensor([[reward]])

        memory.push(state, action.to('cpu'), next_state, reward)

        state = next_state

        if len(memory) > hparams.train_start:
            transitions = memory.sample(hparams.batch_size)
            batch = Transition(*zip(*transitions))

            states = torch.cat(batch.state).to(device)
            actions = torch.cat(batch.action).to(device)
            rewards = torch.cat(batch.reward).to(device)

            non_terminal_mask = torch.tensor([state is not None for state in batch.next_state],
                                             device=device, dtype=torch.bool).unsqueeze(1)
            non_terminal_next_states = torch.cat([state for state in batch.next_state if state is not None]).to(device)

            action_values = q_network(states).gather(1, actions)

            next_state_action_values = torch.zeros_like(action_values)
            next_state_action_values[non_terminal_mask] = target_network(non_terminal_next_states).max(1)[0].detach()
            target_action_values = (next_state_action_values * hparams.gamma) + rewards

            loss = F.smooth_l1_loss(action_values, target_action_values)

            optimizer.zero_grad()
            loss.backward()
            for param in q_network.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()

            if (global_step + 1) % hparams.target_update == 0:
                target_network.load_state_dict(q_network.state_dict())

        global_step += 1
        if done:
            break
    if (episode + 1) % 20 == 0:
        print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(global_step, episode, t, total_reward))
env.close()

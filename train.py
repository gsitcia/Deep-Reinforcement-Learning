import itertools
import argparse

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym

# from comet_ml import Experiment

import wrappers
from networks import DQN
from replay import ReplayMemory, Transition

# experiment = Experiment(project_name="dqn")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
hparams = argparse.Namespace(batch_size=32,
                             n_episodes=1000000,
                             gamma=0.99,
                             eps_start=1,
                             eps_end=0.1,
                             eps_time=1000000,
                             target_update=5000,
                             lr=2.5e-4,
                             momentum=0.95,
                             train_start=10000,
                             memory_size=1000000,
                             render=True)

# make sure we can sample a batch
assert hparams.train_start >= hparams.batch_size


# Epsilon Greedy Policy
def select_action(q, state, global_step):
    eps_start = hparams.eps_start
    eps_end = hparams.eps_end
    eps_time = hparams.eps_time

    # Linear Epsilon Decay
    if global_step < eps_time:
        epsilon = eps_start - (eps_start - eps_end) * global_step / eps_time
    else:
        epsilon = eps_end

    sample = random.random()

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

# Optimizer
optimizer = torch.optim.RMSprop(q_network.parameters(), lr=hparams.lr, momentum=hparams.momentum)

# Environment
env = gym.make("Breakout-v0")
env = wrappers.make_env(env)

global_step = 0
train_started = False
for episode in range(hparams.n_episodes):
    # Initialize the State
    obs = env.reset()
    state = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0)
    total_reward = 0.0
    for t in itertools.count():
        if hparams.render:
            env.render()

        # Select Action
        action = select_action(q_network, state, global_step)

        # Environment Step
        obs, reward, done, info = env.step(action)

        # Next State (None if the episode is over)
        next_state = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0) if not done else None

        # Log Total Episode Reward
        total_reward += reward
        reward = torch.tensor([[reward]])

        # Add (s, a, r, s') to the Replay Buffer
        memory.push(state, action.to('cpu'), next_state, reward)

        # Update Current State
        state = next_state

        if len(memory) > hparams.train_start:
            # Note the start of training
            if not train_started:
                torch.save(torch.tensor(global_step), "train_start_step.txt")
                torch.save(torch.tensor(episode), "train_start_episode.txt")
                # TODO: Log this
                train_started = True

            # Sample a Batch
            transitions = memory.sample(hparams.batch_size)
            batch = Transition(*zip(*transitions))  # trick to transpose data

            # Convert Batch to a Torch Tensor with Batch Dimension 0
            states = torch.cat(batch.state).to(device)
            actions = torch.cat(batch.action).to(device)
            rewards = torch.cat(batch.reward).to(device)

            # Generate a Mask for Non-Terminal States
            non_terminal_mask = torch.tensor([state is not None for state in batch.next_state],
                                             device=device, dtype=torch.bool).unsqueeze(1)
            # All Non-Terminal (s')
            non_terminal_next_states = torch.cat([state for state in batch.next_state if state is not None]).to(device)

            # Compute q(s)
            action_values = q_network(states).gather(1, actions)

            # Set q(s') to 0 and fill in for non-terminal (s')
            next_state_action_values = torch.zeros_like(action_values)
            next_state_action_values[non_terminal_mask] = target_network(non_terminal_next_states).max(1)[0].detach()

            # Compute Targets
            target_action_values = (next_state_action_values * hparams.gamma) + rewards

            # Huber Loss
            loss = F.smooth_l1_loss(action_values, target_action_values)

            # Compute Gradients
            optimizer.zero_grad()
            loss.backward()

            # Clip Gradients
            for param in q_network.parameters():
                param.grad.data.clamp_(-1, 1)

            # Update Parameters
            optimizer.step()

            # Update Target Network
            if (global_step + 1) % hparams.target_update == 0:
                target_network.load_state_dict(q_network.state_dict())

        global_step += 1
        if done:
            break

    # Log to Console
    if (episode + 1) % 20 == 0:
        print(f"Total steps: {global_step + 1} \t Episode: {episode + 1}/{t + 1} \t Total reward: {total_reward}")

    # Save Model
    if (episode + 1) % 100 == 0:
        torch.save(q_network.state_dict(), f"checkpoints/ep_{episode + 1}.pt")
env.close()

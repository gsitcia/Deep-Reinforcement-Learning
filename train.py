import itertools
import argparse

import torch

import gym

import wrappers
from DQN import *

import sys
import time

if sys.stdout.isatty():
    prefix = '\n\x1b[A'
else:
    prefix = ''

# Hyperparameters
hparams = argparse.Namespace(batch_size=32,
                             n_episodes=100000,
                             gamma=0.99,
                             eps_start=1,
                             eps_end=0.1,
                             eps_time=1000000,
                             target_update=10000,
                             lr=2.5e-4,
                             momentum=0.95,
                             train_start=10000,
                             memory_size=500000,
                             load_current=False)

# make sure we can sample a batch
assert hparams.train_start >= hparams.batch_size

if __name__ == "__main__":
    # Environment
    env = gym.make("BreakoutNoFrameskip-v0") # We'll do that
    env = wrappers.make_env(env)

    network = RND(env, hparams)

    if hparams.load_current:
        network.Q.load_state_dict(torch.load("current.pt",map_location=torch.device('cpu')))
        network.T.load_state_dict(torch.load("current.pt",map_location=torch.device('cpu')))

    last = 0
    for i in range(hparams.train_start):
        network.sample_step(1,1)
        network.steps = 0
        if sys.stdout.isatty() and 50*i//hparams.train_start != last:
            last += 1
            sys.stdout.write('.')
            sys.stdout.flush()

    print(prefix+'Starting training'.ljust(50))

    t = time.time()

    reward = 0
    loss = 0
    for step in itertools.count(1):
        reward += network.sample_step()

        if step%4 == 0:
            loss += network.optimize()

        if step%hparams.target_update == 0:
            network.T.load_state_dict(network.Q.state_dict())

        if step%1000 == 0:
            print(prefix+'Saving checkpoint in "checkpoints/step_%d.pt"         '%step)
            torch.save(network.Q.state_dict(), 'checkpoints/step_%d.pt'%step)
            torch.save(network.Q.state_dict(), 'current.pt')
        if step%50 == 0:
            t = time.time()-t
            print(
                prefix+'Total steps:',str(step).ljust(9),
                'Episode:',str(network.n_episodes).ljust(9),
                'Fps:',str(int(50/t)).ljust(4),
                'Reward:',str(reward).ljust(4),
                'Average loss:',loss.item()/50)
            loss = 0
            reward = 0
            t = time.time()
        elif sys.stdout.isatty():
            sys.stdout.write('.')
            sys.stdout.flush()
        if network.n_episodes >= hparams.n_episodes:
            break
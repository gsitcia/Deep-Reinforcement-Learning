import torchvision
import torch
import gym
import subprocess as sp

from networks import *
from wrappers import make_env
import random

import sys

import pyglet

path = "checkpoints/%s.pt"%sys.argv[1]

q = OCNN()
#q.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
q.eval()
env = gym.make('BreakoutNoFrameskip-v0')
env = make_env(env,noop_max=10) # sometimes it no-ops until the ball is unsaveable

obs = env.reset()
action = 0

kold = None

def update(dt):
    global obs, action, kold
    env.render()
    k = q(torch.cat(obs).unsqueeze(0))
    kk = k.tolist()
    if kold != kk:
        print(*('- '[i>0]+str(round(abs(i),3)).ljust(5,'0') for i in kk[0]))
        kold = kk
    if random.random() < 0.9:
        if random.random() < 0.05:
            action = env.action_space.sample()
            #print('Random',action)
        else:
            action = k.max(1)[1].item()
            #print('Chosen',action)
    obs,_,done,_ = env.step(action)
    if done:
        obs = env.reset()

pyglet.clock.schedule_interval(update, 1./60.)
try:
    pyglet.app.run()
except:
    env.close()
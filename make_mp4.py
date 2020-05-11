import torchvision
import torch
import gym
import subprocess as sp

from networks import *
from wrappers import make_env
import random

import sys

@torch.no_grad()
def save_video(env,net,f,func=lambda a,b:b[0]):
	cmd_out = ['ffmpeg',
	           '-f', 'image2pipe',
	           '-vcodec', 'png',
	           '-r', '60',  # FPS 
	           '-i', '-',  # Indicated input comes from pipe 
	           '-vcodec', 'png',
	           '-qscale', '0',
	           f]

	done = False

	pipe = sp.Popen(cmd_out, stdin=sp.PIPE)

	obs = env.reset()
	state = torch.cat(obs).unsqueeze(0)
	games = 0
	action = net(state).max(1)[1].item()
	while games < 10:
		im = func(net,obs)
		torchvision.transforms.ToPILImage()(im).save(pipe.stdin,'PNG')
		obs,_,done,_ = env.step(action)
		if done:
			games += 1
			obs = env.reset()
		state = torch.cat(obs).unsqueeze(0)
		if random.random() < 0.9:
			action = env.action_space.sample() if random.random() < 0.05 else net(state).max(1)[1].item()

	pipe.stdin.close()
	pipe.wait()

	if pipe.returncode != 0:
	    raise sp.CalledProcessError(pipe.returncode, cmd_out)

if __name__ == "__main__":
	path = "checkpoints/%s.pt"%sys.argv[1]

	q = OCNN()
	q.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
	q.eval()
	env = gym.make('BreakoutNoFrameskip-v0')
	env = make_env(env)

	save_video(env,q,'%s.mp4'%sys.argv[1])

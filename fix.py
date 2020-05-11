import torch
import torch.nn as nn
from networks import *
import gym
from wrappers import make_env
import sys

def hook_layers(net,visualization=None):
    if visualization is None:
        visualization = {}
    def hook_fn(m,i,o):
        visualization[m] = i
    for name, layer in net._modules.items():
        if isinstance(layer, nn.Sequential):
            hook_layers(layer, visualization)
        else:
            layer.register_forward_hook(hook_fn)
    return visualization

def modify(t,i):
    t[i] += torch.randn_like(t[i])*0.05

if __name__ == "__main__":
    path = "checkpoints/%s.pt"%sys.argv[1]

    q = OCNN()
    q.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
    q.eval()
    env = gym.make('BreakoutNoFrameskip-v0')
    env = make_env(env)

    d = hook_layers(q)
    
    obs = env.reset()
    done = False

    R = 0
    c = 0

    c1 = 0
    c2 = 0
    c3 = 0

    for i in range(1000):
        state = torch.cat(obs).unsqueeze(0)
        action = q(state).max(1)[1].item()
        obs,reward,done,_ = env.step(action)
        R += reward
        c += 1

        c1 += d[q.conv2][0][0].std(dim=[1,2])
        c2 += d[q.conv3][0][0].std(dim=[1,2])
        c3 += d[q.flatten][0][0].std(dim=[1,2])

    print(R)
    n = 0
    for C,l in zip([c1,c2,c3],[q.conv1,q.conv2,q.conv3]):
        for i in range(C.shape[0]):
            if C[i] == 0:
                modify(l.weight,i)
                modify(l.bias,i)
                n += 1
    torch.save(q.state_dict(),"checkpoints/%s-fixed.pt"%sys.argv[1])
    print('Saved in checkpoints/%s-fixed.pt'%sys.argv[1])
    print(n,'changes made')
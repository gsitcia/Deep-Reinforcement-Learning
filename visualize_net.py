import torch
import torch.nn as nn
from make_mp4 import save_video
import sys
from networks import *
import gym
from wrappers import make_env

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

def blit(dest, src, loc):
    pos = [i if i >= 0 else None for i in loc]
    neg = [-i if i < 0 else None for i in loc]
    target = dest[[slice(i,None) for i in pos]]
    src = src[[slice(i, j) for i,j in zip(neg, target.shape)]]
    target[[slice(None, i) for i in src.shape]] = src
    return dest
# https://stackoverflow.com/questions/28676187/numpy-blit-copy-part-of-an-array-to-another-one-with-a-different-size

def normalize(a):
    m = (torch.sigmoid(2*(a-0.5*(a.min()+a.max()))/(a.max()-a.min()+0.01))*255).type(torch.uint8)
    #m = (255*(a-a.min())/(a.max()-a.min()+0.01)).type(torch.uint8)
    return m
    #torch.stack([255-m,m])

def make_frame(net, d, i):
    for r in range(4):
        k = normalize(d[net.conv1][0][0][r])
        blit(i, k, (2+88*r, 2))
    for r in range(16):
        for c in range(2):
            k = normalize(d[net.conv2][0][0][r*2+c])
            blit(i, k, (1+22*r,89+22*c))
    for r in range(32):
        for c in range(2):
            k = normalize(d[net.conv3][0][0][r*2+c])
            blit(i, k, (1+11*r,134+11*c))
    for r in range(32):
        for c in range(2):
            k = normalize(d[net.flatten][0][0][r*2+c])
            blit(i, k, (33+9*r,157+9*c))
    return i

if __name__ == "__main__":
    path = "checkpoints/%s.pt"%sys.argv[1]

    q = OCNN()
    #q.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
    q.eval()
    env = gym.make('BreakoutNoFrameskip-v0')
    env = make_env(env)

    d = hook_layers(q)
    i = torch.zeros(352,176,dtype=torch.uint8)*128
    w = torch.zeros(352,176,dtype=torch.uint8)*255

    def get_image(net,_):
        global d,i
        i = make_frame(net,d,i)
        return i#torch.stack([w,i,i])

    save_video(env,q,'%s.mp4'%sys.argv[1],get_image)

import sys
pthname_lst = [x for x in sys.argv if x.endswith(".pth")]
if(len(sys.argv) < 2 or len(pthname_lst) != 1):
    print("python3 test_dqn_pong.py model.pth [-g]")
    exit()
pthname = pthname_lst[0]
use_gui = "-g" in sys.argv

from Wrapper.layers import *
from Wrapper.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import math, random
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


USE_CUDA = torch.cuda.is_available()
from dqn import QLearner, compute_td_loss, ReplayBuffer

env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)

num_frames = 1000000
batch_size = 32
gamma = 0.99

replay_initial = 10000
replay_buffer = ReplayBuffer(100000)
model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
model.load_state_dict(torch.load(sys.argv[1], map_location='cpu'))
model.eval()
if USE_CUDA:
    model = model.cuda()
    print("Using cuda")

model.load_state_dict(torch.load(pthname,map_location='cpu'))

env.seed(1)
state = env.reset()
done = False

games_won = 0

while not done:
    if use_gui:
        env.render()

    action = model.act(state, 0)

    state, reward, done, _ = env.step(action)
    
    if reward != 0:
        print(reward)
    if reward == 1:
        games_won += 1

print("Games Won: {}".format(games_won))
try:
    sys.exit(0)
except:
    pass

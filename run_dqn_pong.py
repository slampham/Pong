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

USE_CUDA = torch.cuda.is_available()
from dqn import QLearner, compute_td_loss, ReplayBuffer

# Set up game
env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)

# CONSTS
num_frames = 1000000    # How many frames to play
batch_size = 32
gamma = 0.99
record_idx = 10000      # FIXME: UNUSED const

replay_initial = 10000  # FIXME: dont know what this is
replay_buffer = ReplayBuffer(100000)                                            # Buffer size
model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)             # Create model
model.load_state_dict(torch.load("model.pth", map_location='cpu'))   # FIXME: maybe change model name back to "model_pretrained.pth"?

target_model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)      # Create target model
target_model.copy_from(model)

# Optimize model's parameters
optimizer = optim.Adam(model.parameters(), lr=0.00001)  # FIXME: Maybe change LR?
if USE_CUDA:
    model = model.cuda()
    target_model = target_model.cuda()
    print("Using cuda")

# Neg exp func. Start exploring then exploiting according to frame_indx
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
    -1. * frame_idx / epsilon_decay)

losses = []
all_rewards = []
episode_reward = 0

state = env.reset()  # Initial state

for frame_idx in range(1, num_frames + 1):  # Each frame in # frames played
    # print("Frame: " + str(frame_idx))

    epsilon = epsilon_by_frame(frame_idx)   # Epsilon decreases as frames played
    action = model.act(state, epsilon)      # if (rand < e) explore. Else action w max(Q-val). action: int

    next_state, reward, done, _ = env.step(action)  # Get env info after taking action. next_state: 2d int. reward: float. done: bool.
    replay_buffer.push(state, action, reward, next_state, done)  # Save next_state onto buffer (note: every frame)

    state = next_state                      # Change to next state
    episode_reward += reward                # Keep adding rewards until goal state

    if done:                                # Goal state
        state = env.reset()                 # Restart game
        all_rewards.append((frame_idx, episode_reward))  # Store episode_reward w frame it ended
        episode_reward = 0

    if len(replay_buffer) > replay_initial:     # If enough frames in replay_buffer (10000)
        loss = compute_td_loss(model, target_model, batch_size, gamma, replay_buffer)
        optimizer.zero_grad()                   # Resets gradient after every mini-batch
        loss.backward()
        optimizer.step()
        losses.append((frame_idx, loss.data.cpu().numpy()))

    if frame_idx % 10000 == 0 and len(replay_buffer) <= replay_initial:  # If frames still needed in replay_buffer
        print('#Frame: %d, preparing replay buffer' % frame_idx)

    if frame_idx % 10000 == 0 and len(replay_buffer) > replay_initial:   # If enough frames in replay_buffer
        print('#Frame: %d, Loss: %f' % (frame_idx, np.mean(losses, 0)[1]))
        print('Last-10 average reward: %f' % np.mean(all_rewards[-10:], 0)[1])

    if frame_idx % 50000 == 0:
        target_model.copy_from(model)       # Copy model's weights onto target after 50000 frames
        torch.save(model.state_dict(), "model.pth")
        print("Saved model")

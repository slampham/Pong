import random
from collections import deque

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class QLearner(nn.Module):
    def __init__(self, env, num_frames, batch_size, gamma, replay_buffer):
        super(QLearner, self).__init__()

        self.batch_size = batch_size
        self.gamma = gamma
        self.num_frames = num_frames
        self.replay_buffer = replay_buffer
        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        self.features = nn.Sequential(              # Convolutional layer
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),    # Input 1st layer
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),                 # In --> 32, Out --> 64
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(                    # Fully connected layer
            nn.Linear(self.feature_size(), 512),    # TODO: what is 'feature_size'?
            nn.ReLU(),
            nn.Linear(512, self.num_actions)        # Output num_actions
        )

    def forward(self, x):
        x = self.features(x)            # Pass in state_transition
        x = x.view(x.size(0), -1)       # Reshape x
        x = self.fc(x)                  # Pass in features onto last layer
        return x        # ret (batch_size=32, num_actions=6). shape = 2

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state, epsilon):  # Neg exp curve. Start exploring then exploit
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), requires_grad=True)
            action = torch.argmax(self.forward(state)).item()  # Given state, write code to get Q value and chosen action
        else:
            action = random.randrange(self.env.action_space.n)
        return action

    def copy_from(self, target):
        self.load_state_dict(target.state_dict())


def compute_td_loss(model, target_model, batch_size, gamma, replay_buffer):
    # Returns 32 in each batch!! (i.e. 32 states, 32 action...)
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)).squeeze(1), requires_grad=True)
    action = Variable(torch.LongTensor(action))  # action / state. 32 actions / batch
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))    # 'done' is float to simplify Qn equation

    Q = model(state.squeeze(1)).gather(dim=1, index=action.view(-1, 1)).flatten()
    Qn = reward + (1 - done) * gamma * target_model(next_state).max(dim=1)[0].detach()
    loss = nn.MSELoss()(Qn, Q)

    return loss


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):    # Done: bool
        state = np.expand_dims(state, 0)                # Expand row-wise
        next_state = np.expand_dims(next_state, 0)      # Expand row-wise
        self.buffer.append((state, action, reward, next_state, done))   # Insert tuple of Q

    def sample(self, batch_size):  # Initially, frame_idx = replay_buffer.len. But, sampling reduces replay_buffer size
        # Randomly sampling data with specific batch size from the buffer
        batch = random.sample(self.buffer, batch_size)  # Batch_size = 32

        state = [state_transition[0] for state_transition in batch]
        action = [state_transition[1] for state_transition in batch]
        reward = [state_transition[2] for state_transition in batch]
        next_state = [state_transition[3] for state_transition in batch]
        done = [state_transition[4] for state_transition in batch]

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gym
import minerl
class DQNAgent(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNAgent, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.vstack(state), action, reward, np.vstack(next_state), done

class DQNLearner:
    def __init__(self, input_size, output_size, gamma=0.99, epsilon_decay=0.995,
                 learning_rate=1e-3, buffer_capacity=10000, batch_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNAgent(input_size, output_size).to(self.device)
        self.target_net = DQNAgent(input_size, output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.policy_net.fc[-1].out_features)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                q_values = self.policy_net(state)
                return torch.argmax(q_values).item()

    def train(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        current_q_values = self.policy_net(state).gather(1, action.unsqueeze(1))
        next_q_values = self.target_net(next_state).max(1)[0].unsqueeze(1)
        target_q_values = reward + (1 - done) * self.gamma * next_q_values

        loss = self.criterion(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(0.1, self.epsilon)

action_names = ['forward','back','left','right','jump']

env = gym.make("MineRLBasaltFindCave-v0")
state_size = 360*640
action_size = 5

learner = DQNLearner(state_size, action_size)
num_episodes = 500
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    while True:
        action = env.action_space.noop()
        action_num = learner.select_action(state)
        # print(action_num)
        action_name = action_names[action_num]
        action[action_name] = 1
        next_state, reward, done, _ = env.step(action)
        learner.replay_buffer.push(state, action, reward, next_state, done)
        learner.train()

        state = next_state
        total_reward += reward

        if done:
            learner.update_target_network()
            learner.decay_epsilon()
            break

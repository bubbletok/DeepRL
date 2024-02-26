import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
class DQN(nn.Module):
    # lr = learning rate
    # input_dims = 입력층 차원
    # fc1_dims = 은닉층 1 차원
    # fc2_dimx = 은닉층 2 차원
    # n_actions = 행동 횟수, 즉 결과층 차원
    # fc1 = 은닉층 1, fc2 = 은닉층 2
    # fc3 = 결과층
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DQN,self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims,self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims,self.n_actions)
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    # 순전파 함수
    def forward(self, state):
        # 입력 -> 은닉층 1
        x = F.relu(self.fc1(state))

        # 은닉층 1 -> 은닉층 2
        x = F.relu(self.fc2(x))

        # 은닉층 2 -> 결과층
        actions = self.fc3(x)

        return actions
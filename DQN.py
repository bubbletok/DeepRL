import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
class DQN(nn.Module):
    # lr = learning rate
    # input_dims = �Է��� ����
    # fc1_dims = ������ 1 ����
    # fc2_dimx = ������ 2 ����
    # n_actions = �ൿ Ƚ��, �� ����� ����
    # fc1 = ������ 1, fc2 = ������ 2
    # fc3 = �����
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
    
    # ������ �Լ�
    def forward(self, state):
        # �Է� -> ������ 1
        x = F.relu(self.fc1(state))

        # ������ 1 -> ������ 2
        x = F.relu(self.fc2(x))

        # ������ 2 -> �����
        actions = self.fc3(x)

        return actions
from DQN import DQN
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size,
                 n_actions, max_mem_size=100000,
                 eps_end=0.01, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.mem_size = max_mem_size
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        # mem_cntr : ?? memeory의 current r?
        self.mem_cntr = 0

        # action spcae define, how?
        # In the video,
        # self.action_space = [i for i in range(n_actions)]
        # it can be like above because each index is matched with specific actoin
        # 0: GO LEFT, 1: GO DOWN, ...
        # But in the minerl, action space is Dictionary.
        # So, for finding a cave, we only use move(forward,back,left,right). And camera if necessary

        # "forward": "Discrete(2)"
        # "back": "Discrete(2)"
        # "left": "Discrete(2)"
        # "right": "Discrete(2)"

        self.action_space = [i for i in range(n_actions)]

        # Q evaluation?
        self.Q_eval = DQN(self.lr,input_dims=input_dims,
                          fc1_dims=256, fc2_dims=256,n_actions=n_actions,)
        
        # replay memory
        # state, new_state
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype={})
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype={})

        # action, reward
        # terminial, which is for saving whether episode is done or not
        self.action_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size,dtype=np.bool)

    def store_replay(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index]= new_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

    def greedy_policy(self,observations):
        state = T.tensor([observations]).to(self.Q_eval.device)
        actions = self.Q_eval.forward(state)
        action = T.argmax(actions).itme()
        return action
    
    def epsilon_greedy(self, observations):
        rand_num = np.random.random()
        if rand_num > self.epsilon:
            action = self.greedy_policy(observations)
        else:
            action = np.random.choice(self.action_space)
        return action
    
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr,self.mem_size)
        batch = np.random.choice(max_mem,self.batch_size,replace=False)

        batch_index = np.arange(self.batch_size,dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = T.tensor(self.action_memory[batch]).to(self.Q_eval.device)

        # q_eval : former Q-value estimation, 즉 이전 q value 측정 값
        q_eval = self.Q_eval.forward(state_batch)[batch_index,action_batch]
        # q_next : Q-value of next state, 다음 상태의 q value  측정 값
        q_next = self.Q_eval.forward(new_state_batch)
        # 아직 다음 상태는 끝나지 않았으므로 0으로 지정
        q_next[terminal_batch] = 0.0

        # Q-target 공식 그대로 적용
        q_target = reward_batch + self.gamma * T.max(q_next,dim=1)[0]

        # Q-loss
        loss = self.Q_eval.loss(q_target,q_eval).to(self.Q_eval.device)
        # loss 역전파??
        loss.backward()
        # optimizer 실행
        self.Q_eval.optimizer.step()

        # epsilon update
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_minelse else self.eps_min
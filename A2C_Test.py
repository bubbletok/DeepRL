import gym
from FindCaveAgent_A2C import Agent
# from utils import plotLearning
import numpy as np

agent = Agent(alpha=0.00001, beta=0.00005)

env = gym.make('LunarLander-v2')
score_history=[]
num_episodes = 2000

for i in range(num_episodes):
    done = False
    score = 0
    observation = env.reset()

    while not done:
        action = agent.choose_actions(observation)
        observation_,reward,done,info = env.step(action)
        agent.learn(observation,action,reward,observation_,done)
        observation = observation_
        score += reward
    
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    print('episode ', i, 'score %.2f average score %.2f ' % \
          (score, avg_score))
    
filename = 'lunar-lander-actor-critic.png'
#plotLearning(score_history, filename=filename, window=100)



# -*- coding: utf-8 -*-
import gym
import numpy as np
import minerl
import random
# from learning.utils import plot_learning_curve

from FindCaveAgent import Agent
# Uncomment to see more logs of the MineRL launch
# import coloredlogs
# coloredlogs.install(logging.DEBUG)


# action_names = ['forward','jump','camera']
# 0 1 2 3 4 5 6 7 8 9
# action // 2 -> 0 1 2 3 4
# action - action//2 -> 0 1

# 0 1 2 3 4 5 6 7
# f f f f j j j j
    
# 8 9 10 11 12 13 14
# c c c c

env = gym.make("MineRLBasaltFindCave-v0")

action_names = ['forward','jump','camera']
# for i in env.action_space.keys():
#     action_names.append(i)

camera_idx = 2

camera_angles = []
for i in range(-10,11,10):
    if i == 0:
        continue
    new_angle = [0,i]
    camera_angles.append(new_angle)
print(len(camera_angles))
# print(action_names)
print(len(action_names))
agent = Agent(gamma=0.99,epsilon=1.0,batch_size=64,n_actions=len(action_names) + len(camera_angles) - 1,
                eps_end=0.01, input_dims=[255], lr=0.003)
scores, eps_history = [], []
n_episodes = 500

def postprocess_obs(obs):
     # Only use image data
    obs = obs['pov'].squeeze().astype(np.float32)
    # Transpose observations to be channel-first (BCHW instead of BHWC)
    # obs = obs.transpose(2, 0, 1)
    # Normalize observations
    obs /= 255.0
    return obs

def take_action(action, num):
    if num >= camera_idx and num < camera_idx + len(camera_angles):
        action_name = action_names[camera_idx]
        action[action_name] = camera_angles[num-camera_idx]
    else:
        if num >= camera_idx + len(camera_angles):
            action_name = action_names[num-(len(camera_angles)-1)]
            action[action_name] = 1
        else:
            action_name = action_names[num]
            action[action_name] = 1
    return action

def train():
    for i in range(n_episodes):
        score = 0.
        done = False
        obs = env.reset()
        while not done:
            # obs = postprocess_obs(obs)
            # In BASALT environments, sending ESC action will end the episode

            # Take a action based on epslion greedy policy
            action = env.action_space.noop()
            action['ESC'] = 0
            # action -> env.step 할때만 쓰이는 실제 action_space : Dict
            # action_num -> DQN에서 쓰일 action_space
            action_num = agent.epsilon_greedy(obs)
            action = take_action(action,action_num)
            # print(action_num)

            new_obs, reward, done, _ = env.step(action)
            # new_obs = postprocess_obs(new_obs)
            score += reward
            # save replay data
            agent.store_replay(obs,action_num,reward,new_obs,done)

            # train agent
            agent.learn()
            obs = new_obs
            env.render()

        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        
        print('episoed ', i , 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)
    env.close()

train()
# x = [i+1 for i in range(n_episodes)]
# filename = 'MineRLBasaltFindCave-v0_LearningHistory.png'
# plot_learning_curve(x,scores,eps_history,filename)
    
# Save the list to a text file
with open('scores.txt', 'w') as file:
    for item in scores:
        file.write(f"{item}\n")

with open('eps_history.txt', 'w') as file:
    for item in eps_history:
        file.write(f"{item}\n")



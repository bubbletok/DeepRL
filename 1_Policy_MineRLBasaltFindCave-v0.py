# -*- coding: utf-8 -*-
import gym
import gym.spaces.utils
import numpy as np
import torch
import minerl
import random
from Policy import Policy
from collections import deque
import matplotlib.pyplot as plt
import torch.optim as optim

# Uncomment to see more logs of the MineRL launch
# import coloredlogs
# coloredlogs.install(logging.DEBUG)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env_id = "MineRLBasaltFindCave-v0"

# Create the env
env = gym.make(env_id)

flatten_observation_space = gym.spaces.utils.flatten_space(env.observation_space['pov'])
# print("Flattened observation shape:", flatten_observation_space)

action_names = []
for i in env.action_space.keys():
    action_names.append(i)

camera_angles = []
for i in range(-180,181,10):
    new_angle = [0,i]
    camera_angles.append(new_angle)

# # Get the state space and action space
s_size = flatten_observation_space.shape[0]
a_size = len(action_names) + len(camera_angles) - 1

print(action_names)
# print(len(action_names),len(camera_angles))
# print(s_size)
# print(a_size)

def reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=100)
    scores = []
    # Line 3 of pseudocode
    for i_episode in range(1, n_training_episodes+1):
        saved_log_probs = []
        rewards = []
        state = env.reset() # TODO: reset the environment
        # Line 4 of pseudocode
        for t in range(max_t):
            # print("state:",state)
            state = state['pov'].reshape(-1)
            # print("new state:",state)
            action, log_prob = policy.act(state) # TODO get the action
            print("Action:", action)
            new_action = env.action_space.noop()
            if(3<=action and action <= 39):
                new_action['camera'] = camera_angles[action-3]
            else:
                if action > 39:
                    new_action[action_names[action - (len(camera_angles) - 1)]] = 1
                else:
                    new_action[action_names[action]] = 1
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(new_action)# TODO: take an env step
            rewards.append(reward)
            env.render()
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # Line 6 of pseudocode: calculate the return
        returns = deque(maxlen=max_t)
        n_steps = len(rewards)
        # Compute the discounted returns at each timestep,
        # as the sum of the gamma-discounted return at time t (G_t) + the reward at time t

        # In O(N) time, where N is the number of time steps
        # (this definition of the discounted return G_t follows the definition of this quantity
        # shown at page 44 of Sutton&Barto 2017 2nd draft)
        # G_t = r_(t+1) + r_(t+2) + ...

        # Given this formulation, the returns at each timestep t can be computed
        # by re-using the computed future returns G_(t+1) to compute the current return G_t
        # G_t = r_(t+1) + gamma*G_(t+1)
        # G_(t-1) = r_t + gamma* G_t
        # (this follows a dynamic programming approach, with which we memorize solutions in order
        # to avoid computing them multiple times)

        # This is correct since the above is equivalent to (see also page 46 of Sutton&Barto 2017 2nd draft)
        # G_(t-1) = r_t + gamma*r_(t+1) + gamma*gamma*r_(t+2) + ...


        ## Given the above, we calculate the returns at timestep t as:
        #               gamma[t] * return[t] + reward[t]
        #
        ## We compute this starting from the last timestep to the first, in order
        ## to employ the formula presented above and avoid redundant computations that would be needed
        ## if we were to do it from first to last.

        ## Hence, the queue "returns" will hold the returns in chronological order, from t=0 to t=n_steps
        ## thanks to the appendleft() function which allows to append to the position 0 in constant time O(1)
        ## a normal python list would instead require O(N) to do this.
        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns)>0 else 0)
            returns.appendleft(gamma*disc_return_t + rewards[t]) # TODO: complete here

        ## standardization of the returns is employed to make training more stable
        eps = np.finfo(np.float32).eps.item()

        ## eps is the smallest representable float, which is
        # added to the standard deviation of the returns to avoid numerical instabilities
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # Line 7:
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        # Line 8: PyTorch prefers gradient descent
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    return scores


findcave_hyperparameters = {
    "h_size": 16,
    "n_training_episodes": 50,
    "n_evaluation_episodes": 10,
    "max_t": 1000,
    "gamma": 1.0,
    "lr": 1e-2,
    "env_id": env_id,
    "state_space": s_size,
    "action_space": a_size,
}

# # Create policy and place it to the device
findcave_policy = Policy(device, findcave_hyperparameters["state_space"], findcave_hyperparameters["action_space"], findcave_hyperparameters["h_size"]).to(device)
findcave_optimizer = optim.Adam(findcave_policy.parameters(), lr=findcave_hyperparameters["lr"])


scores = reinforce(findcave_policy,
                   findcave_optimizer,
                   findcave_hyperparameters["n_training_episodes"],
                   findcave_hyperparameters["max_t"],
                   findcave_hyperparameters["gamma"],
                   100)

# # Save the list to a text file
# with open('scores.txt', 'w') as file:
#     for item in scores:
#         file.write(f"{item}\n")

# with open('eps_history.txt', 'w') as file:
#     for item in eps_history:
#         file.write(f"{item}\n")

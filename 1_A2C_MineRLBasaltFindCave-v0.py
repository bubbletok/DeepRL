# -*- coding: utf-8 -*-
import gym
import numpy as np
import minerl
import random

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

env_id = "MineRLBasaltFindCave-v0"

# Create the env
env = gym.make(env_id)

# Get the state space and action space
# s_size = env.observation_space.shape
# a_size = env.action_space

print("_____OBSERVATION SPACE_____ \n")
# print("The State Space is: ", s_size)
print("Sample observation", env.observation_space.sample()) # Get a random observation

print("\n _____ACTION SPACE_____ \n")
# print("The Action Space is: ", a_size)
print("Action Space Sample", env.action_space.sample()) # Take a random action

env = DummyVecEnv([lambda: env])

env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

model = A2C(policy = "MultiInputPolicy",
            env = env,
            verbose=1)


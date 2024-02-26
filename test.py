import gym
import minerl
import random
# Uncomment to see more logs of the MineRL launch
# import coloredlogs
# coloredlogs.install(logging.DEBUG)
action_names = ['forward','left','jump']

env = gym.make("MineRLBasaltFindCave-v0")
done = False
obs = env.reset()
print(env.observation_space)
score = 0
while not done:
    # Take a random action
    action = env.action_space.noop()
    for name in action_names:
        action[name] = random.randint(0,1)
    # In BASALT environments, sending ESC action will end the episode
    # Lets not do that
    obs, reward, done, _ = env.step(action)
    score += reward
    print(reward)
    env.render()
#print(score)
env.close()



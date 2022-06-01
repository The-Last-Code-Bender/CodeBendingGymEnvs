import gym
from gym_codebending.envs import gregworld, cartpole_pymunk
from gym_codebending.envs.pygame_utils import colour_constants
import pygame
import numpy as np


env = gym.make('CartPolePyMunk-v0')
print(env.observation_space_table)
#env.render()


for i in range(5):
    env.reset()
    while True:
        env.human_play()
        next_obs, reward, done, info = env.step(0)

        env.render()

        if done:
            break
import logging
from gym.envs.registration import register
logger = logging.getLogger(__name__)
register(id="GregWorld-v0", entry_point="gym_codebending.envs:GregWorld", max_episode_steps=200)
register(id="CartPolePyMunk-v0", entry_point="gym_codebending.envs:CartPolePyMunk", max_episode_steps=1000)

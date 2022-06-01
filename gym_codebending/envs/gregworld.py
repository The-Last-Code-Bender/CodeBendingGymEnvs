import numpy as np
import pygame
from typing import Tuple, Optional, Union
import pathlib
import gym
from gym.core import ActType, ObsType

UP = np.array([0, -1])
RIGHT = np.array([1, 0])
DOWN = np.array([0, 1])
LEFT = np.array([-1, 0])
NO_OP = np.array([0, 0])

WIDTH, HEIGHT = 800, 800
BLACK = (0, 0, 0)
FPS = 60
CURRENT_PATH = pathlib.Path(__file__).parent.absolute()
ground_tile_path = CURRENT_PATH/"gregworld_assets/sprites/ground_tile.png"
wall_tile_path = CURRENT_PATH/"gregworld_assets/sprites/wall_tile.png"
start_tile_path = CURRENT_PATH/"gregworld_assets/sprites/start_tile.png"
goal_tile_path = CURRENT_PATH/"gregworld_assets/sprites/goal_tile.png"
greg_tile_path = CURRENT_PATH/"gregworld_assets/sprites/greg.png"


class GregWorld(gym.Env):

    def __init__(self, world='7x7-simple', delay=0):

        file = open(CURRENT_PATH/f"gregworld_assets/gregworld_maps/{world}.csv")
        env = np.loadtxt(file, delimiter=';', dtype=object)
        base_shape = env.shape

        walls = np.array(['w' for _ in range(base_shape[0])])

        env = np.c_[env, walls]
        env = np.c_[walls, env]

        walls = np.array([['w' for _ in range(base_shape[1] + 2)]])

        env = np.r_[env, walls]
        env = np.r_[walls, env]

        self.env = env.T

        state_counter = 0
        self.observation_table = dict()

        for x, row in enumerate(env):
            for y, col in enumerate(row):
                if col != 'w':
                    self.observation_table[(x, y)] = state_counter
                    state_counter += 1

        start_pos = np.where(env == 's')
        start_pos = int(start_pos[0]), int(start_pos[1])
        end_pos = np.where(env == 't')
        end_pos = int(end_pos[0]), int(end_pos[1])

        self.start_pos = np.array(start_pos)
        self.end_pos = np.array(end_pos)

        self.current_pos = start_pos

        self.observation_space = gym.spaces.Discrete(len(self.observation_table))
        self.action_space = gym.spaces.Discrete(5)

        self.action_table = {
            0: NO_OP,
            1: LEFT,
            2: RIGHT,
            3: UP,
            4: DOWN
        }

        # Rendering configuration
        self.delay = delay
        sprite_dims_x = int(WIDTH / self.env.shape[0])
        sprite_dims_y = int(HEIGHT / self.env.shape[1])

        self.GROUND_SPRITE = pygame.transform.scale(pygame.image.load(ground_tile_path), (sprite_dims_x, sprite_dims_y))
        self.WALL_SPRITE = pygame.transform.scale(pygame.image.load(wall_tile_path), (sprite_dims_x, sprite_dims_y))
        self.START_SPRITE = pygame.transform.scale(pygame.image.load(start_tile_path), (sprite_dims_x, sprite_dims_y))
        self.GOAL_SPRITE = pygame.transform.scale(pygame.image.load(goal_tile_path), (sprite_dims_x, sprite_dims_y))
        self.GREG = pygame.transform.scale(pygame.image.load(greg_tile_path), (sprite_dims_x, sprite_dims_y))

        self.sprite_dims_x = sprite_dims_x
        self.sprite_dims_y = sprite_dims_y

        self.render_mode = False
        self.window = None
        self.clock = None

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> Union[ObsType, tuple[ObsType, dict]]:
        if self.render_mode:
            pygame.time.wait(self.delay)
        self.current_pos = self.start_pos
        return self.observation_table[tuple(self.current_pos)]

    def render(self, mode="human"):
        self.render_mode = True

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()
        events = pygame.event.get()

        self.window.fill(BLACK)
        for x, row in enumerate(self.env):
            for y, col in enumerate(row):
                sprite_to_draw = self.GROUND_SPRITE
                if col == 'w':
                    sprite_to_draw = self.WALL_SPRITE
                elif col == 's':
                    sprite_to_draw = self.START_SPRITE
                elif col == 't':
                    sprite_to_draw = self.GOAL_SPRITE
                self.window.blit(sprite_to_draw, (x * self.sprite_dims_x, y * self.sprite_dims_y))

        self.window.blit(self.GREG, (self.current_pos[0] * self.sprite_dims_x, self.current_pos[1] * self.sprite_dims_y))
        self.clock.tick(FPS)

        pygame.display.flip()
        return events

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:

        action = self.action_table[action]
        next_pos = self.current_pos + action

        if self.env[tuple(next_pos)] != 'w':
            self.current_pos = next_pos

        reward = -1
        done = False

        if (self.current_pos == self.end_pos).all():
            reward = 0
            done = True

        current_state = self.observation_table[tuple(self.current_pos)]
        if self.render_mode:
            pygame.time.wait(self.delay)
        return current_state, reward, done, {}

    def sample(self):
        return self.action_space.sample()
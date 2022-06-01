import sys
from typing import Optional, Union, Tuple

import numpy as np

from .pygame_utils.rendering import GameWindow
from .pygame_utils import colour_constants as cc
import pymunk
import pygame
import gym
from gym.core import ObsType, ActType

WIDTH = 1200
HEIGHT = 600
FPS = 60
CART_MAX_VELOCITY = 1000
TRACK_MIN_X = 50
TRACK_MAX_X = WIDTH - TRACK_MIN_X
GRAVITY = 500
CART_START_POS = (WIDTH / 2, 400)
POLE_START_POS = (WIDTH / 2, 300)
POLE_SHAPE = (25, 300)
CART_SHAPE = (100, 50)
N_STATE_VALUES = 3
ACTION_TABLE = {
    0: (0, 0),
    1: (-200, 0),
    2: (200, 0)
}


def limit_velocity(body, gravity, damping, dt):
    pymunk.Body.update_velocity(body, gravity, damping, dt)
    l = body.velocity.length
    if l > CART_MAX_VELOCITY:
        scale = CART_MAX_VELOCITY / l
        body.velocity = body.velocity * scale


def get_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()


class CartPolePyMunk(gym.Env):

    def get_interp_state(self):
        pole_angle = np.interp(self.pole.angle, (-self.rad_limit, self.rad_limit), (-1, 1))
        pole_angular_velocity = np.interp(self.pole.angular_velocity, (-self.angular_limit, self.angular_limit),
                                          (-1, 1))

        # Only the x position and velocity matter
        cart_velocity = np.interp(self.cart.velocity[0], (-CART_MAX_VELOCITY, CART_MAX_VELOCITY), (-1, 1))
        cart_pos = np.interp(self.cart.position[0], (TRACK_MIN_X, TRACK_MAX_X), (-1, 1))

        next_state = [
            pole_angle,
            pole_angular_velocity,
            #cart_velocity,
            cart_pos
        ]

        if self.discrete:
            for s in range(len(next_state)):
                next_state[s] = np.digitize(next_state[s], self.bins)

        return next_state

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        self.apply_action(action)
        self.space.step(self.dt)

        next_state = self.get_interp_state()

        done = False
        reward = 1 - abs(np.interp(self.pole.angle, (-self.rad_limit, self.rad_limit), (-1, 1)))
        reward += 1 - abs(np.interp(self.pole.angular_velocity, (-self.angular_limit, self.angular_limit), (-1, 1)))

        if abs(self.pole.angle) > self.rad_limit:
            done = True
            reward = 0

        return next_state, reward, done, {}

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> Union[
        ObsType, tuple[ObsType, dict]]:
        self.cart.position = CART_START_POS
        self.cart.velocity = (0, 0)
        self.pole.position = POLE_START_POS
        self.pole.velocity = (0, 0)
        self.pole.angle = 0
        self.pole.angular_velocity = 0

        self.cart.apply_impulse_at_local_point((np.random.randint(-70, 70), 0))
        self.space.step(self.dt)
        next_state = self.get_interp_state()

        return next_state

    def render(self, mode="human"):
        if self.window is None:
            self.render_mode = True
            self.window = GameWindow(WIDTH, HEIGHT, FPS)

        get_events()
        self.window.render(self.space)

    def apply_action(self, action):
        if (abs(self.cart.position[0] - TRACK_MIN_X) < 0.1) or (abs(self.cart.position[0] - TRACK_MAX_X) < 0.1):
            action = 0
        self.cart.apply_impulse_at_local_point(ACTION_TABLE[action])

    def human_play(self):
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    action = 1
                if event.key == pygame.K_d:
                    action = 2
        self.apply_action(action)

    def __init__(self, discrete=True, n_bins=50):

        self.action_space = gym.spaces.Discrete(3)

        self.rad_limit = np.deg2rad(45)
        self.angular_limit = 3

        self.discrete = discrete
        if self.discrete:
            self.bins = np.linspace(-10, 10, num=n_bins)
            self.observation_space = gym.spaces.Box(low=0, high=n_bins, shape=(1, N_STATE_VALUES))
            self.observation_space_table = [n_bins + 1] * N_STATE_VALUES

        # PyMunk
        self.space = pymunk.Space()
        self.space.gravity = (0, GRAVITY)

        self.cart = pymunk.Body()
        self.cart.position = CART_START_POS
        self.cart_shape = pymunk.Poly.create_box(self.cart, size=CART_SHAPE)
        self.cart_shape.mass = 1
        self.cart_shape.color = cc.DARKORANGE1.get_rgba()
        self.cart.velocity_func = limit_velocity

        self.pole = pymunk.Body()
        self.pole.position = POLE_START_POS
        self.pole_shape = pymunk.Poly.create_box(self.pole, size=POLE_SHAPE)
        self.pole_shape.mass = 0.1
        self.pole_shape.color = cc.AQUAMARINE1.get_rgba()

        self.cartpole_joint = pymunk.PivotJoint(self.cart, self.pole, (0, 0), (0, CART_START_POS[1] - POLE_SHAPE[1]))
        self.cartpole_joint.collide_bodies = False

        move_joint = pymunk.GrooveJoint(self.space.static_body, self.cart, (TRACK_MIN_X, 400), (TRACK_MAX_X, 400), (0, 0))
        self.space.add(self.cart, self.cart_shape, move_joint, self.pole, self.pole_shape, self.cartpole_joint)

        # Rendering
        pygame.init()
        self.render_mode = False
        self.dt = 1 / FPS
        self.draw_options = None
        self.clock = None
        self.window = None

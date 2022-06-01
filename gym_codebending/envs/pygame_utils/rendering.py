import pygame
from pymunk.pygame_util import DrawOptions


def get_game_window(width, height):
    pygame.init()
    window = pygame.display.set_mode((width, height))
    return window


class GameWindow:
    def __init__(self, width=600, height=600, fps=60):
        self.window = get_game_window(width, height)
        self.draw_options = DrawOptions(self.window)
        self.clock = pygame.time.Clock()
        self.fps = fps

    def render(self, space):
        self.window.fill("white")
        space.debug_draw(self.draw_options)
        pygame.display.flip()
        self.clock.tick(self.fps)

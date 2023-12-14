import pygame

class Object_UI:
    def __init__(self):
        self.obstacles = pygame.sprite.Group()
        self.agents = pygame.sprite.Group()
        self.targets = pygame.sprite.Group()
        self.enviroments = pygame.display.get_surface()
    def run(self):
        pass
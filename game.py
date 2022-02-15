#!/usr/bin/python

import sys
import random
import numpy as np
import os
import pygame as pg
from pygame.locals import *

WIN_WIDTH = 768
WIN_HEIGHT = 512

main_dir = os.path.split(os.path.abspath(__file__))[0]
data_dir = os.path.join(main_dir, "resources")
directions = ["up", "left", "right", "down"]

def main():
    # Initialise screen
    pg.init()
    screen = pg.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pg.display.set_caption('Skillshot Dodger')
    obstacles = []
    # pg.time.set_timer(USEREVENT + 2, random.randrange(2000, 3500))

    # Fill background
    background = pg.image.load("resources/background.jpg")
    background = pg.transform.scale(background, screen.get_size())
    background = background.convert()

    # Blit everything to the screen
    screen.blit(background, [0, 0])
    pg.display.flip()

    player = Player()
    allsprites = pg.sprite.RenderPlain((player))
    clock = pg.time.Clock()

    fireball = Fireball()
    fireball.draw(screen)

    obstacles.append(fireball)
    pg.key.set_repeat(2)

    pg.time.set_timer(USEREVENT + 2, random.randrange(200, 350)) # determines how often we generate a fireball
    # Event loop
    while True:
        # dt = clock.tick(120)
        clock.tick(60)

        for obstacle in obstacles:
            # move the obstacle
            if obstacle.direction == "up":
                obstacle.y -= obstacle.vel
            elif obstacle.direction == "down":
                obstacle.y += obstacle.vel
            elif obstacle.direction == "left":
                obstacle.x -= obstacle.vel
            else:
                obstacle.x += obstacle.vel

        for event in pg.event.get():
            # generate a new fireball
            if event.type == USEREVENT+2:
                obstacles.append(Fireball())

            if event.type == QUIT:
                return
            pressed_keys = pg.key.get_pressed()

            key_direction = np.array([0,0])
            if pressed_keys[K_LEFT]: key_direction[0] = -1
            if pressed_keys[K_RIGHT]: key_direction[0] = 1 
            if pressed_keys[K_DOWN]: key_direction[1] = 1
            if pressed_keys[K_UP]: key_direction[1] = -1

            # key_direction *= dt # ?? - keeps frames consistent but its very fast
            # TODO: no fractional movement
            # Idea - keep track of actual position and round before displaying to screen
            # norm = np.linalg.norm(key_direction)
            # if norm > 0:
            #     key_direction = key_direction / norm
            #     print(key_direction)

            x, y = player.rect.topleft
            player.rect.topleft = (x + key_direction[0], y + key_direction[1])
            print(key_direction)

        allsprites.update()
        for obstacle in obstacles:
            obstacle.update()

        # Draw Everything
        pg.display.update()
        screen.blit(background, (0, 0))
        allsprites.draw(screen)

        for obstacle in obstacles:
            obstacle.draw(screen)

        pg.display.flip()

def load_image(name, colorkey=None, scale=1):
    fullname = os.path.join(data_dir, name)
    image = pg.image.load(fullname)

    size = image.get_size()
    size = (int(size[0] * scale), int(size[1] * scale))
    image = pg.transform.scale(image, size)

    image = image.convert()
    if colorkey is not None:
        if colorkey == -1:
            colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, pg.RLEACCEL)
    return image, image.get_rect()

class Fireball(pg.sprite.Sprite):
    # need to define some rational pictures so we can have fireballs in many directions
    # rotate = [pg.image.load(os.path.join('resources', 'SAW0.PNG')), pg.image.load(os.path.join('resources', 'SAW1.PNG')),
    #           pg.image.load(os.path.join('resources', 'SAW2.PNG')), pg.image.load(os.path.join('resources', 'SAW3.PNG'))]
    def __init__(self):
        # generate random location for the fireball on edge of screen
        pg.sprite.Sprite.__init__(self)  # call Sprite initializer
        i = np.random.choice([0, 1])
        if i % 2 == 0:
            self.x = np.random.randint(0, WIN_WIDTH)
            self.y = np.random.choice([0, WIN_HEIGHT])
        else:
            self.x = np.random.choice([0, WIN_HEIGHT])
            self.y = np.random.randint(0, WIN_WIDTH)
        self.width = 1
        self.height = 1
        self.rotateCount = 0
        self.vel = np.random.randint(10, 18)
        self.image, self.rect = load_image("fireball.jpeg", scale=0.3)
        self.direction = self.getDirection(self.x, self.y)

    def getDirection(self, x, y):
        if x == 0:
            return "right"
        if x == WIN_WIDTH:
            return "left"
        if y == 0:
            return "down"
        if y == WIN_HEIGHT:
            return "up"

    def draw(self, screen):
        print("sheesh")
        self.hitbox = (self.x + 10, self.y + 10, 28, 10)  # defines the hitbox
        pg.draw.rect(screen, (255, 0, 0), self.hitbox, 2)
        screen.blit(self.image, (self.x, self.y)) # not sure why this is so choppy lol

class Player(pg.sprite.Sprite):
    def __init__(self):
        pg.sprite.Sprite.__init__(self)  # call Sprite initializer
        self.image, self.rect = load_image("player-sprite.gif", scale=0.15)

    def move_down(self):
        x, y = self.rect.topleft
        self.rect.topleft = (x, y+1)

    def move_up(self):
        x, y = self.rect.topleft
        self.rect.topleft = (x, y-1)

    def move_right(self):
        x, y = self.rect.topleft
        self.rect.topleft = (x+1, y)

    def move_left(self):
        x, y = self.rect.topleft
        self.rect.topleft = (x-1, y)


    def update(self):
        """move the fist based on the mouse position"""
        # pos = pg.mouse.get_pos()
        # self.rect.topleft = pos
        # self.rect.move_ip(self.fist_offset)
        # if self.punching:
        #     self.rect.move_ip(15, 25)



if __name__ == '__main__':
    main()

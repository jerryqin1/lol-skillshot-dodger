#!/usr/bin/python

import sys
import random
import math
import numpy as np
import os
import pygame as pg
from pygame.locals import *

WIN_WIDTH = 768
WIN_HEIGHT = 512

main_dir = os.path.split(os.path.abspath(__file__))[0]
data_dir = os.path.join(main_dir, "resources")
directions = ['N', 'E', 'W', 'S', 'NW', 'NE', 'SW', 'SE']



### TODO:
###      1. code refactoring: create wrapper for main() (hold it in a Game class) and allow for environemnt reset and runthrough
###      2. fine tune collision parameters / hitboxes
###      3.

def main():
    # Initialise screen
    pg.init()
    screen = pg.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pg.display.set_caption('Skillshot Dodger')
    font = pg.font.SysFont("comic sans", 40)

    obstacles = []
    score = 0

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

    pg.time.set_timer(USEREVENT + 2, random.randrange(200, 300)) # determines how often we generate a fireball
    # Event loop
    while True:
        # dt = clock.tick(120)
        clock.tick(60)
        score += 1

        player_c_x, player_c_y = player.rect.topleft
        player_c_x += player.rect.width / 2
        player_c_y += player.rect.height / 2
        # player_radius = math.sqrt((player.rect.width / 2) ** 2 +(player.rect.height / 2) ** 2)
        # player_radius = 34

        for obstacle in obstacles:
            # move the obstacle
            obstacle.x += obstacle.x_vel
            obstacle.y += obstacle.y_vel

            fireball_radius = int (obstacle.rect.width / 2)
            obs_c_x = obstacle.x + fireball_radius
            obs_c_y = obstacle.y + fireball_radius

            distance = math.dist((player_c_x, player_c_y), (obs_c_x, obs_c_y))

            # if distance < fireball_radius + player_radius:
            if distance < 32:
                print("I got hit")
                print("Final score:", score)
                return

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

        scoretext = font.render("Score: " + str(score), True, (255, 255, 255), (0, 0, 0))
        screen.blit(scoretext, (5, 5))

        # Draw Everything
        pg.display.update()
        screen.blit(background, (0, 0))
        allsprites.draw(screen)

        for obstacle in obstacles:
            obstacle.draw(screen)

        # pg.display.flip()

    print('Final score: ' + str(score))

def load_image(name, colorkey=None, scale=1):
    fullname = os.path.join(data_dir, name)
    image = pg.image.load(fullname)
    image.set_colorkey((0,0,0))

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
    def __init__(self):
        # generate random location for the fireball on edge of screen
        pg.sprite.Sprite.__init__(self)  # call Sprite initializer
        i = np.random.choice([0, 1])
        if i % 2 == 0:
            self.x = np.random.randint(0, WIN_WIDTH)
            self.y = np.random.choice([0, WIN_HEIGHT])
        else:
            self.x = np.random.choice([0, WIN_WIDTH])
            self.y = np.random.randint(0, WIN_HEIGHT)
        self.rotateCount = 0
        self.image, self.rect = load_image("fireball.png", scale=0.15)
        self.direction = self.getDirection()
        self.x_vel, self.y_vel = self.getVel()

    def getVel(self):
        n = len(self.direction)
        x_vel = 0
        y_vel = 0

        if self.direction[0] == 'N':
            y_vel = -np.random.randint(10, 12)
        elif self.direction[0] == 'S':
            y_vel = np.random.randint(10, 12)
        elif self.direction[0] == 'W':
            x_vel = -np.random.randint(10, 12)
        else:
            x_vel = np.random.randint(10, 12)

        if n == 2:
            x_vel = -np.random.randint(10, 12) if self.direction[1] == 'W' else np.random.randint(10, 12)

        return x_vel, y_vel

    def getDirection(self):
        if self.x == 0:
            if self.y < (WIN_HEIGHT / 2):
                return np.random.choice(['SE', 'SE', 'E'])
            else:
                return np.random.choice(['NE', 'NE', 'E'])
        elif self.x == WIN_WIDTH:
            if self.y < (WIN_HEIGHT / 2):
                return np.random.choice(['SW', 'SW', 'W'])
            else:
                return np.random.choice(['NW', 'NW', 'W'])

        if self.y == 0:
            if self.x < (WIN_WIDTH / 2):
                return np.random.choice(['SE', 'SE', 'S'])
            else:
                return np.random.choice(['SW', 'SW', 'S'])
        elif self.y == WIN_HEIGHT:
            if self.x < (WIN_WIDTH / 2):
                return np.random.choice(['NE', 'NE', 'N'])
            else:
                return np.random.choice(['NW', 'NW', 'N'])



    def draw(self, screen):
        print("sheesh")
        # self.hitbox = (self.x + 10, self.y + 10, 28, 10)  # defines the hitbox
        # pg.draw.rect(screen, (255, 0, 0), self.hitbox, 2)
        screen.blit(self.image, (self.x, self.y)) # not sure why this is so choppy lol

class Player(pg.sprite.Sprite):
    def __init__(self):
        pg.sprite.Sprite.__init__(self)  # call Sprite initializer
        # self.image, self.rect = load_image("player-sprite.gif", scale=0.15)
        self.image, self.rect = load_image("poro_icon.png", scale=1.7)
        self.rect.topleft = (WIN_WIDTH / 2, WIN_HEIGHT / 2)

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

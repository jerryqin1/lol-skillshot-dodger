#!/usr/bin/python

import sys
import random
import math
import numpy as np
import os
import pygame as pg
from pygame.locals import *

simple = True

def check_bump(x_pos, y_pos, sprite_width, sprite_height):
    if x_pos <= 0:
        x_pos = BUMP_DIST
    elif x_pos >= WIN_WIDTH - sprite_width:
        x_pos = WIN_WIDTH - BUMP_DIST - sprite_width

    if y_pos <= 0:
        y_pos = BUMP_DIST
    elif y_pos >= WIN_HEIGHT - sprite_height:
        y_pos = WIN_HEIGHT - BUMP_DIST - sprite_height

    return x_pos, y_pos

main_dir = os.path.split(os.path.abspath(__file__))[0]
data_dir = os.path.join(main_dir, "resources")

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
        # self.image, self.rect = load_image("fireball4.png", scale=0.044)
        if not simple:
            self.image, self.rect = load_image("fireball4.png", scale=0.044)
        else:
            self.image, self.rect = load_image("simple_red1.png", scale=0.032)
        self.direction = self.getDirection()
        self.x_vel, self.y_vel = self.getVel()

    def getVel(self):
        n = len(self.direction)
        x_vel = 0
        y_vel = 0

        MAX_SPD = 14
        MID_SPD = 7
        MIN_SPD = 1

        if self.direction[0] == 'N':
            y_vel = -np.random.randint(MIN_SPD, MAX_SPD)
        elif self.direction[0] == 'S':
            y_vel = np.random.randint(MIN_SPD, MAX_SPD)
        elif self.direction[0] == 'W':
            x_vel = -np.random.randint(MIN_SPD, MAX_SPD)
        else:
            x_vel = np.random.randint(MIN_SPD, MAX_SPD)

        if n == 2:
            if y_vel < 7:
                x_vel = -np.random.randint(MAX_SPD - 2, MAX_SPD) if self.direction[1] == 'W' else np.random.randint(MAX_SPD - 2, MAX_SPD)
            else:
                x_vel = -np.random.randint(MIN_SPD, MID_SPD) if self.direction[1] == 'W' else np.random.randint(MIN_SPD, MID_SPD)

        return x_vel, y_vel

    def getDirection(self):
        if self.x == 0:
            if self.y < (WIN_HEIGHT / 2):
                return 'SE'
            else:
                return 'NE'
        elif self.x == WIN_WIDTH:
            if self.y < (WIN_HEIGHT / 2):
                return 'SW'
            else:
                return 'NW'

        if self.y == 0:
            if self.x < (WIN_WIDTH / 2):
                return 'SE'
            else:
                return 'SW'
        elif self.y == WIN_HEIGHT:
            if self.x < (WIN_WIDTH / 2):
                return 'NE'
            else:
                return 'NW'



    def draw(self, screen):
        # print("sheesh")
        # self.hitbox = (self.x + 10, self.y + 10, 28, 10)  # defines the hitbox
        # pg.draw.rect(screen, (255, 0, 0), self.hitbox, 2)
        screen.blit(self.image, (self.x, self.y)) # not sure why this is so choppy lol


class Player(pg.sprite.Sprite):
    def __init__(self):
        pg.sprite.Sprite.__init__(self)  # call Sprite initializer
        # self.image, self.rect = load_image("player-sprite.gif", scale=0.15)
        # self.image, self.rect = load_image("poro_icon.png", scale=1.7)
        if not simple:
            self.image, self.rect = load_image("poro_icon.png", scale=1.7)
        else:
            self.image, self.rect = load_image("simple_green.png", scale=0.028)
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


pg.init()

WIN_WIDTH = 768
WIN_HEIGHT = 512
BUMP_DIST = 3
FPS = 60

ACTION_MAP = {
    0 : (0, 0),
    1 : (0, 1),
    2 : (1, 1),
    3 : (1, 0),
    4 : (1, -1),
    5 : (0, -1),
    6 : (-1, -1),
    7 : (-1, 0),
    8: (-1, 1),
}

screen = pg.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
clock = pg.time.Clock()


# Fill background
if not simple:
    background = pg.image.load("resources/background.jpg")
else:
    background = pg.image.load("resources/white_color.png")

# background = pg.image.load("resources/background.jpg")
background = pg.transform.scale(background, screen.get_size())
background = background.convert()


class GameState:
    def __init__(self):
        self.player = Player()
        self.allsprites = pg.sprite.RenderPlain((self.player))
        self.score = 0
        self.obstacles = []

        pg.time.set_timer(USEREVENT + 2, random.randrange(150, 200)) # determines how often we generate a fireball

    def frame_step(self, action):
        action = np.argmax(action)
        # dt = clock.tick(120)
        # clock.tick(60)

        terminal = False

        pg.event.pump()

        # TODO - fix this
        self.score += 1

        player_c_x, player_c_y = self.player.rect.topleft
        player_c_x += self.player.rect.width / 2
        player_c_y += self.player.rect.height / 2

        for obstacle in self.obstacles:
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
                print("Final score:", self.score)
                terminal = True
                self.__init__()
                break

        # TODO - obs gen
        for event in pg.event.get():
            # generate a new fireball
            if event.type == USEREVENT+2:
                self.obstacles.append(Fireball())

        key_direction = ACTION_MAP[action]

        x, y = self.player.rect.topleft
        # TODO - change hardcode
        x, y = check_bump(x + key_direction[0], y + key_direction[1], 40, 32)
        self.player.rect.topleft = (x, y)

        self.allsprites.update()
        for obstacle in self.obstacles:
            obstacle.update()

        # scoretext = font.render("Score: " + str(score), True, (255, 255, 255), (0, 0, 0))
        # screen.blit(scoretext, (5, 5))

        # Draw Everything
        pg.display.update()
        screen.blit(background, (0, 0))
        self.allsprites.draw(screen)

        for obstacle in self.obstacles:
            if obstacle.x <= -20 or obstacle.y <= -20 or obstacle.x >= WIN_WIDTH + 20 or obstacle.y >= WIN_HEIGHT + 20:
                self.obstacles.pop(self.obstacles.index(obstacle))
            else:
                obstacle.draw(screen)

        image_data = pg.surfarray.array3d(pg.display.get_surface())
        clock.tick(FPS)
        return image_data, 1, terminal, self.score

    def reset(self):
        # pg.init()
        # screen = pg.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        pg.time.set_timer(USEREVENT + 2, random.randrange(150, 200))  # determines how often we generate a fireball

        # Fill background
        background = pg.image.load("resources/background.jpg")
        background = pg.transform.scale(background, screen.get_size())
        background = background.convert()
        self.score = 0
        self.player = Player()
        self.allsprites = pg.sprite.RenderPlain((self.player))
        self.obstacles = []
        print("woo")

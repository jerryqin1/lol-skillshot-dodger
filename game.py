#!/usr/bin/python

import sys
import argparse
import random
import math
import numpy as np
import os
import pygame as pg
from pygame.locals import *

parser = argparse.ArgumentParser(description='Choose gameplay mode.')
# parser.add_argument()

WIN_WIDTH = 768
WIN_HEIGHT = 512
BUMP_DIST = 3

simple = True

main_dir = os.path.split(os.path.abspath(__file__))[0]
data_dir = os.path.join(main_dir, "resources")
directions = ['N', 'E', 'W', 'S', 'NW', 'NE', 'SW', 'SE']


### TODO:
###      1. code refactoring: create wrapper for main() (hold it in a Game class) and allow for environemnt reset and runthrough
###      2. fine tune collision parameters / hitboxes
###      3.

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


def main():
    # Initialise screen
    pg.init()
    screen = pg.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pg.display.set_caption('Skillshot Dodger')
    font = pg.font.SysFont("comic sans", 40)

    obstacles = []
    score = 0

    # Fill background
    if not simple:
        background = pg.image.load("resources/background.jpg")
    else:
        background = pg.image.load("resources/white_color.png")
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

    pg.time.set_timer(USEREVENT + 2, random.randrange(150, 200))  # determines how often we generate a fireball
    # Event loop
    while True:
        # dt = clock.tick(120)
        clock.tick(60)
        score += (10 / 6)

        player_c_x, player_c_y = player.rect.topleft
        player_c_x += player.rect.width / 2
        player_c_y += player.rect.height / 2
        # player_radius = math.sqrt((player.rect.width / 2) ** 2 +(player.rect.height / 2) ** 2)
        # player_radius = 34

        for obstacle in obstacles:
            # move the obstacle
            obstacle.x += obstacle.x_vel
            obstacle.y += obstacle.y_vel

            fireball_radius = int(obstacle.rect.width / 2)
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
            if event.type == USEREVENT + 2:
                obstacles.append(Fireball())

            if event.type == QUIT:
                return
            pressed_keys = pg.key.get_pressed()

            key_direction = np.array([0, 0])
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
            x, y = check_bump(x + key_direction[0], y + key_direction[1], 40, 32)
            player.rect.topleft = (x, y)

            # print(key_direction)

        allsprites.update()
        for obstacle in obstacles:
            obstacle.update()

        scoretext = font.render("Score: " + str(score), True, (255, 255, 255), (0, 0, 0))
        screen.blit(scoretext, (5, 5))

        # Draw Everything
        pg.display.update()
        screen.blit(background, (0, 0))
        allsprites.draw(screen)

        next_obstacles = []
        for obstacle in obstacles:
            if obstacle.x <= -20 or obstacle.y <= -20 or obstacle.x >= WIN_WIDTH + 20 or obstacle.y >= WIN_HEIGHT + 20:
                pass
            else:
                next_obstacles.append(obstacle)
                obstacle.draw(screen)

        obstacles = next_obstacles


def load_image(name, colorkey=None, scale=1):
    fullname = os.path.join(data_dir, name)
    image = pg.image.load(fullname)
    image.set_colorkey((0, 0, 0))

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
        self.speed = 14
        if i % 2 == 0:
            self.x = np.random.randint(0, WIN_WIDTH)
            self.y = np.random.choice([0, WIN_HEIGHT])
        else:
            self.x = np.random.choice([0, WIN_WIDTH])
            self.y = np.random.randint(0, WIN_HEIGHT)
        self.rotateCount = 0
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

        MAX_SPD = self.speed
        MID_SPD = self.speed / 2
        MIN_SPD = 0

        if self.direction[0] == 'N':
            y_vel = -np.random.randint(MIN_SPD, MAX_SPD)
        elif self.direction[0] == 'S':
            y_vel = np.random.randint(MIN_SPD, MAX_SPD)
        elif self.direction[0] == 'W':
            x_vel = -np.random.randint(MIN_SPD, MAX_SPD)
        else:
            x_vel = np.random.randint(MIN_SPD, MAX_SPD)

        # make everything the same speed indifferent of direction
        if n == 2:
            x_speed = int(math.sqrt(self.speed**2 - y_vel**2))
            x_vel = -x_speed if self.direction[1] == 'W' else x_speed
        elif y_vel == 0:
            x_vel = -MAX_SPD if self.direction[1] == 'W' else MAX_SPD
        elif x_vel == 0:
            y_vel = -MAX_SPD if self.direction[1] == 'N' else MAX_SPD

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
        screen.blit(self.image, (self.x, self.y))  # not sure why this is so choppy lol


class Player(pg.sprite.Sprite):
    def __init__(self):
        pg.sprite.Sprite.__init__(self)  # call Sprite initializer
        # self.image, self.rect = load_image("player-sprite.gif", scale=0.15)
        if not simple:
            self.image, self.rect = load_image("poro_icon.png", scale=1.7)
        else:
            self.image, self.rect = load_image("simple_green.png", scale=0.028)
        self.rect.topleft = (WIN_WIDTH / 2, WIN_HEIGHT / 2)

    def move_down(self):
        x, y = self.rect.topleft
        self.rect.topleft = (x, y + 1)

    def move_up(self):
        x, y = self.rect.topleft
        self.rect.topleft = (x, y - 1)

    def move_right(self):
        x, y = self.rect.topleft
        self.rect.topleft = (x + 1, y)

    def move_left(self):
        x, y = self.rect.topleft
        self.rect.topleft = (x - 1, y)

    def update(self):
        """move the fist based on the mouse position"""
        # pos = pg.mouse.get_pos()
        # self.rect.topleft = pos
        # self.rect.move_ip(self.fist_offset)
        # if self.punching:
        #     self.rect.move_ip(15, 25)


if __name__ == '__main__':
    main()

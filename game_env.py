import gym
from gym import spaces
import numpy as np
import os
import pygame as pg
from pygame.locals import USEREVENT
import math
import random
import cv2

WIN_WIDTH = 256
WIN_HEIGHT = 256
BUMP_DIST = 3
FPS = 60
NUM_ACTIONS = 9

simple = True
os.environ["SDL_VIDEODRIVER"] = 'dummy'


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


# main_dir = os.path.split(os.path.abspath(__file__))[0]
data_dir = 'resources'


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
        # generate random location for the fireball on edge of self.screen
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
        self.speed = 14
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
            x_speed = int(math.sqrt(self.speed ** 2 - y_vel ** 2))
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
        screen.blit(self.image, (self.x, self.y))  # not sure why this is so choppy lol


class Player(pg.sprite.Sprite):
    def __init__(self):
        pg.sprite.Sprite.__init__(self)  # call Sprite initializer
        if not simple:
            self.image, self.rect = load_image("poro_icon.png", scale=1.7)
        else:
            self.image, self.rect = load_image("simple_green.png", scale=0.028)
        self.speed = 10
        self.rect.topleft = (WIN_WIDTH / 2, WIN_HEIGHT / 2)

        self.ACTION_MAP = {
            0: (0, 0),
            1: (0, 1),
            2: (0, -1),
            3: (1, 1),
            4: (1, 0),
            5: (1, -1),
            6: (-1, -1),
            7: (-1, 0),
            8: (-1, 1),
        }


class GameEnv(gym.Env):
    def __init__(self):
        super(GameEnv, self).__init__()
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=255, shape=(80, 80), dtype=np.uint8)

        # self.renderval = os.environ["SDL_VIDEODRIVER"]
        # os.environ["SDL_VIDEODRIVER"] = 'dummy'
        pg.init()
        self.screen = pg.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        self.clock = pg.time.Clock()
        self.background = pg.image.load("resources/background.jpg") if not simple else pg.image.load(
            "resources/white_color.png")
        self.background = pg.transform.scale(self.background, self.screen.get_size())
        self.background = self.background.convert()
        self.player = Player()
        self.allsprites = pg.sprite.RenderPlain((self.player))
        self.vel = 9
        self.obstacles = []
        self.mintime = 150
        self.maxtime = 200
        pg.time.set_timer(USEREVENT + 2,
                          random.randrange(self.mintime, self.maxtime))  # determines how often we generate a fireball

    def step(self, action):
        # action = np.argmax(action)
        pg.event.pump()

        player_c_x, player_c_y = self.player.rect.topleft
        player_c_x += self.player.rect.width / 2
        player_c_y += self.player.rect.height / 2

        for obstacle in self.obstacles:
            # move the obstacle
            obstacle.x += obstacle.x_vel
            obstacle.y += obstacle.y_vel

            fireball_radius = int(obstacle.rect.width / 2)
            obs_c_x = obstacle.x + fireball_radius
            obs_c_y = obstacle.y + fireball_radius

            distance = math.dist((player_c_x, player_c_y), (obs_c_x, obs_c_y))

            # if distance < fireball_radius + player_radius:
            if distance < 32:
                # print("I got hit")
                # print("Final score:", self.score)
                image_data = pg.surfarray.array3d(pg.display.get_surface())
                x_t = cv2.cvtColor(cv2.resize(image_data, (80, 80)), cv2.COLOR_BGR2GRAY)
                return x_t, -100, True, {}

        for event in pg.event.get():
            # generate a new fireball
            if event.type == USEREVENT + 2:
                self.obstacles.append(Fireball())

        key_direction = self.get_vel(self.player.ACTION_MAP[action])

        x, y = self.player.rect.topleft
        # TODO - change hardcode
        x, y = check_bump(x + key_direction[0], y + key_direction[1], 40, 32)
        self.player.rect.topleft = (x, y)

        self.allsprites.update()
        for obstacle in self.obstacles:
            obstacle.update()

        # scoretext = font.render("Score: " + str(score), True, (255, 255, 255), (0, 0, 0))
        # self.screen.blit(scoretext, (5, 5))

        # Draw Everything
        pg.display.update()
        self.screen.blit(self.background, (0, 0))
        self.allsprites.draw(self.screen)

        for obstacle in self.obstacles:
            if obstacle.x <= -1 or obstacle.y <= -1 or obstacle.x >= WIN_WIDTH + 1 or obstacle.y >= WIN_HEIGHT + 1:
                self.obstacles.pop(self.obstacles.index(obstacle))
            else:
                obstacle.draw(self.screen)

        image_data = pg.surfarray.array3d(pg.display.get_surface())
        x_t = cv2.cvtColor(cv2.resize(image_data, (80, 80)), cv2.COLOR_BGR2GRAY)
        self.clock.tick(FPS)
        return x_t, 10, False, {}

    def reset(self):
        self.obstacles = []
        # pg.time.set_timer(USEREVENT + 2, random.randrange(150, 200))  # determines how often we generate a fireball

        # Fill background
        self.background = pg.image.load("resources/background.jpg") if not simple else pg.image.load(
            "resources/white_color.png")
        self.background = pg.transform.scale(self.background, self.screen.get_size())
        self.background = self.background.convert()
        self.score = 0
        self.player = Player()
        self.allsprites = pg.sprite.RenderPlain((self.player))

        image_data = pg.surfarray.array3d(pg.display.get_surface())
        x_t = cv2.cvtColor(cv2.resize(image_data, (80, 80)), cv2.COLOR_BGR2GRAY)

        return x_t

    def render(self, mode="human"):
        # os.environ["SDL_VIDEODRIVER"] = self.renderval
        pass

    def unrender(self):
        os.environ["SDL_VIDEODRIVER"] = 'dummy'

    def get_vel(self, key_direction):
        if key_direction[0] != 0 and key_direction[1] != 0:
            return key_direction[0] * int(math.sqrt(self.player.speed) / 2), key_direction[1] * int(
                math.sqrt(self.player.speed) / 2)
        else:
            return key_direction[0] * self.player.speed, key_direction[1] * self.player.speed

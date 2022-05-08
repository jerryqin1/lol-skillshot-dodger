import gym
from gym import spaces
import numpy as np
import os
import pygame as pg
from pygame.locals import USEREVENT
import math
import random
import cv2

# Common parameters
WIN_WIDTH = 600
WIN_HEIGHT = 600
BUMP_DIST = 3
FPS = 60
NUM_ACTIONS = 9

simple = True


# Function to ensure player does not move off screen
def check_bump(x_pos, y_pos, sprite_width, sprite_height):
    bump_const = 60
    if x_pos <= bump_const:
        x_pos = bump_const
    elif x_pos >= WIN_WIDTH - sprite_width - bump_const:
        x_pos = WIN_WIDTH - BUMP_DIST - sprite_width - bump_const

    if y_pos <= bump_const:
        y_pos = bump_const
    elif y_pos >= WIN_HEIGHT - sprite_height - bump_const:
        y_pos = WIN_HEIGHT - BUMP_DIST - sprite_height - bump_const

    return x_pos, y_pos


data_dir = 'resources'


# Load image from given filepath
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


# CLass that representst the Fireball sprite
class Fireball(pg.sprite.Sprite):
    def __init__(self):
        # generate random location for the fireball on edge of self.screen
        pg.sprite.Sprite.__init__(self)  # call Sprite initializer

        # Determines where Fireball will be initalized
        i = np.random.choice([0, 1])
        if i % 2 == 0:
            self.x = np.random.randint(0, WIN_WIDTH)
            self.y = np.random.choice([0, WIN_HEIGHT])
        else:
            self.x = np.random.choice([0, WIN_WIDTH])
            self.y = np.random.randint(0, WIN_HEIGHT)
        self.rotateCount = 0

        # Choose fireball image depending on if simple mode is enabled
        if not simple:
            self.image, self.rect = load_image("fireball4.png", scale=0.044)
        else:
            self.image, self.rect = load_image("simple_red1.png", scale=0.032)

        # Set direction and speed
        self.direction = self.getDirection()
        self.speed = 14
        self.x_vel, self.y_vel = self.getVel()

    # Returns velocity of x and y
    def getVel(self):
        n = len(self.direction)
        x_vel = 0
        y_vel = 0

        MAX_SPD = self.speed
        MIN_SPD = 1

        # Give fireballs randoms speed depending on its direction
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

    # Returns direction of Fireball given its starting position
    # Direction will try to maximize fireball's time on the screen
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
        screen.blit(self.image, (self.x, self.y))


# Class that represents player
class Player(pg.sprite.Sprite):
    def __init__(self):
        pg.sprite.Sprite.__init__(self)  # call Sprite initializer

        # Load sprites depending on if simple mode is enabled
        if not simple:
            self.image, self.rect = load_image("poro_icon.png", scale=1.7)
        else:
            self.image, self.rect = load_image("simple_green.png", scale=0.028)

        # Set player speed and start location
        self.speed = 10
        self.rect.topleft = (WIN_WIDTH / 2, WIN_HEIGHT / 2)

        # Map that maps a number to a player's X and Y direction
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


# Environment for skillshot game - inherits from a Gym Environment
class GameEnv(gym.Env):
    def __init__(self):
        # Call super constructor
        super(GameEnv, self).__init__()

        # Set action_space and observation_space -- required for gym environment
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=255, shape=(80, 80, 1), dtype=np.uint8)

        # Initialize pygame and set screen and game properties
        pg.init()
        self.screen = pg.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        self.clock = pg.time.Clock()
        self.background = pg.image.load("resources/background.jpg") if not simple else pg.image.load(
            "resources/white_color.png")
        self.background = pg.transform.scale(self.background, self.screen.get_size())
        self.background = self.background.convert()

        # Create player and player speed
        self.player = Player()
        self.allsprites = pg.sprite.RenderPlain((self.player))
        self.vel = 9

        # Set up obstacles and their container
        self.obstacles = []
        self.max_obstacles = 6
        self.mintime = 150
        self.maxtime = 200

        # determines how often we generate a fireball
        pg.time.set_timer(USEREVENT + 2,
                          random.randrange(self.mintime, self.maxtime))

    # Steps to the next state - required for Gym environments
    def step(self, action):
        pg.event.pump()

        # Get center of player
        player_c_x, player_c_y = self.player.rect.topleft
        player_c_x += self.player.rect.width / 2
        player_c_y += self.player.rect.height / 2

        # Move every obstacle
        for obstacle in self.obstacles:
            # move the obstacle
            obstacle.x += obstacle.x_vel
            obstacle.y += obstacle.y_vel

            # Get center of fireball
            fireball_radius = int(obstacle.rect.width / 2)
            obs_c_x = obstacle.x + fireball_radius
            obs_c_y = obstacle.y + fireball_radius

            # Calculate distance
            distance = math.dist((player_c_x, player_c_y), (obs_c_x, obs_c_y))

            # Collision occurs
            if distance < 32:
                # Get image data, return corresponding reward, and indicate a terminal state (collision)
                image_data = pg.surfarray.array3d(pg.display.get_surface())
                x_t = cv2.cvtColor(cv2.resize(image_data, (80, 80), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2GRAY)
                x_t = x_t[:, :, np.newaxis]
                return x_t, -100, True, {}

        # Ensure max_obstacle obstalces are on screen always
        while len(self.obstacles) < self.max_obstacles:
            self.obstacles.append(Fireball())

        # Get velocity of player given their action
        key_direction = self.get_vel(self.player.ACTION_MAP[action])

        # Move player
        x, y = self.player.rect.topleft
        x, y = check_bump(x + key_direction[0], y + key_direction[1], 40, 32)
        self.player.rect.topleft = (x, y)

        # Update all sprites
        self.allsprites.update()
        for obstacle in self.obstacles:
            obstacle.update()

        # Draw Everything
        pg.display.update()
        self.screen.blit(self.background, (0, 0))
        self.allsprites.draw(self.screen)

        # Remove obstacles when they exit the screen
        tmp_obstacles = []
        for obstacle in self.obstacles:
            if obstacle.x <= -1 or obstacle.y <= -1 or obstacle.x >= WIN_WIDTH + 1 or obstacle.y >= WIN_HEIGHT + 1:
                continue
            else:
                tmp_obstacles.append(obstacle)
                obstacle.draw(self.screen)

        self.obstacles = tmp_obstacles

        # Preprocess image and return image data, episodic reward, and indicate state is not terminal
        image_data = pg.surfarray.array3d(pg.display.get_surface())
        x_t = cv2.cvtColor(cv2.resize(image_data, (80, 80), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2GRAY)
        x_t = x_t[:, :, np.newaxis]
        self.clock.tick(FPS)
        return x_t, 10, False, {}

    # Reset state to a new state ready to be played from the start again - requried for Gym environments
    def reset(self):
        # Reset obstalces
        self.obstacles = []

        # Fill background
        self.background = pg.image.load("resources/background.jpg") if not simple else pg.image.load(
            "resources/white_color.png")
        self.background = pg.transform.scale(self.background, self.screen.get_size())
        self.background = self.background.convert()
        self.score = 0
        self.player = Player()
        self.allsprites = pg.sprite.RenderPlain((self.player))

        # Preprocess frame and return
        image_data = pg.surfarray.array3d(pg.display.get_surface())
        x_t = cv2.cvtColor(cv2.resize(image_data, (80, 80), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2GRAY)
        x_t = x_t[:, :, np.newaxis]

        return x_t

    # Render next time step - not useful - required for Gym environments
    def render(self, mode="human"):
        os.environ["SDL_VIDEODRIVER"] = 'windib'
        pass

    # Debugging function
    def unrender(self):
        os.environ["SDL_VIDEODRIVER"] = 'dummy'

    # Get the velcotiy of x and y given the direction player is moving in
    def get_vel(self, key_direction):
        if key_direction[0] != 0 and key_direction[1] != 0:
            return key_direction[0] * int(math.sqrt(self.player.speed) / 2), key_direction[1] * int(
                math.sqrt(self.player.speed) / 2)
        else:
            return key_direction[0] * self.player.speed, key_direction[1] * self.player.speed

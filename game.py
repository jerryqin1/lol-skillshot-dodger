#!/usr/bin/python

try:
    import sys
    #import random
    #import math
    import os
    #import getopt
    import pygame as pg
    from socket import *
    from pygame.locals import *
except ImportError, err:
    print "couldn't load module. %s" % (err)
    sys.exit(2)

main_dir = os.path.split(os.path.abspath(__file__))[0]
data_dir = os.path.join(main_dir, "resources")

def main():
    # Initialise screen
    pg.init()
    screen = pg.display.set_mode((1500, 1000))
    pg.display.set_caption('Skillshot dodger')

    # Fill background
    background = pg.Surface(screen.get_size())
    background = background.convert()
    background.fill((250, 250, 250))

    # Blit everything to the screen
    screen.blit(background, (0, 0))
    pg.display.flip()

    player = Player()
    allsprites = pg.sprite.RenderPlain((player))
    clock = pg.time.Clock()

    pg.key.set_repeat(2)
    # Event loop
    while 1:
        clock.tick(60)
        for event in pg.event.get():
            if event.type == QUIT:
                return
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_DOWN:
                    player.move_down()
                elif event.key == pg.K_UP:
                    player.move_up()
                elif event.key == pg.K_RIGHT:
                    player.move_right()
                elif event.key == pg.K_LEFT:
                    player.move_left()

        allsprites.update()

        # Draw Everything
        screen.blit(background, (0, 0))
        allsprites.draw(screen)
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

class Player(pg.sprite.Sprite):
    def __init__(self):
        pg.sprite.Sprite.__init__(self)  # call Sprite initializer
        self.image, self.rect = load_image("player-sprite.gif", scale=0.25)

    def move_down(self):
        x, y = self.rect.topleft
        self.rect.topleft = (x, y+5)

    def move_up(self):
        x, y = self.rect.topleft
        self.rect.topleft = (x, y-5)

    def move_right(self):
        x, y = self.rect.topleft
        self.rect.topleft = (x+5, y)

    def move_left(self):
        x, y = self.rect.topleft
        self.rect.topleft = (x-5, y)


    def update(self):
        """move the fist based on the mouse position"""
        # pos = pg.mouse.get_pos()
        # self.rect.topleft = pos
        # self.rect.move_ip(self.fist_offset)
        # if self.punching:
        #     self.rect.move_ip(15, 25)



if __name__ == '__main__':
    main()

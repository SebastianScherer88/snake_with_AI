# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 21:27:13 2018

@author: bettmensch
"""

import pygame as pg
import sys
from settings import *

def draw_tile(screen,
              position,
              color):
    '''Helper function that takes:
        - the screen on which to draw.
        - the position tuple (x,y) of the tile to draw, in the tiled coordinate system
        - the color of the tile.'''
        
    # --- convert to pixel units
    #   position
    position_pixel = (position[0] * TILE_WIDTH, position[1] * TILE_HEIGHT)
    
    #   get positional rect
    positional_rect = pg.Rect(position_pixel[0],
                              position_pixel[1],
                              TILE_WIDTH,
                              TILE_HEIGHT)
    
    # --- draw colored tile to screen
    pg.draw.rect(screen, color, positional_rect)
    
    return

class Snake(object):
    '''Represents the snake on a WINDOW_WIDTH x WINDOW_HEIGHT tiled board.'''
    
    def __init__(self,
                 length=SNAKE_INIT_LENGTH,
                 head_position_x = WINDOW_WIDTH // 2,
                 head_position_y = WINDOW_HEIGHT // 2,
                 direction=LEFT,
                 color = BLUE):
        
        # length of snake in tiles
        self.length = length
        
        # direction of movement
        self.direction = direction
                
        # body of snake  = list of snake's body's tiles in tile coordinate system
        self.body = [(head_position_x,head_position_y)] + [(head_position_x + i,head_position_y) for i in range(1,length)]
        
        # color
        self.color = color
        
    def update(self):
        '''Gets called once every frame. Moves the snake by one tile.'''
        
        # --- header positions
        #    initialize storage for new head position
        current_head_position = self.body[0]
        
        #    get current head position
        new_head_position = [None,None]
        
        # --- get new head position based on current direction
        cur_dir = self.direction
        
        #   moving LEFT / RIGHT
        if cur_dir == LEFT:
            new_head_position[0] = current_head_position[0] - 1
        elif cur_dir == RIGHT:
            new_head_position[0] = current_head_position[0] + 1
        else:
            new_head_position[0] = current_head_position[0]
            
        #   moving UP / DOWN
        if cur_dir == UP:
            new_head_position[1] = current_head_position[1] - 1
        elif cur_dir == DOWN:
            new_head_position[1] = current_head_position[1] + 1
        else:
            new_head_position[1] = current_head_position[1]
            
        # --- move forward
        #   insert new head
        self.body.insert(0,new_head_position)
        
        #   delete old tail
        self.body.pop(-1)
        
    def draw(self,
             screen):
        '''Gets called once every frame. Draws the snake to the screen.'''
        
        for snake_tile_position in self.body:
            draw_tile(screen,
                      snake_tile_position,
                      self.color)
            
class Snake_With_AI(object):
    '''Game class. Represent a game of snake.'''
    
    def __init__(self,
                 looping = True):
        self.board_color = WHITE
        self.food_color = RED
        self.clock = pg.time.Clock()
        self.screen = None
        self.snake = None
        self.food = None
        self.looping = looping
        
        # --- set up pygame window
        #   start up pygame
        pg.init()
        # intialize graphic window
        self.screen = pg.display.set_mode((WINDOW_WIDTH_PIXELS,WINDOW_HEIGHT_PIXELS))
        #   set caption
        pg.display.set_caption("SNAKE WITH AI")
        # fill background
        self.screen.fill(self.board_color)
        
    def start(self):
        '''Called to start a new game.'''
        
        # if game is looping, this loop causes infinite games
        while True:
            # --- draw game sprites
            #   snake
            self.snake = Snake()
            self.snake.draw(self.screen)
        
            # first food
            pass
        
            # manual stop control flag
            manual_quit = False
        
            # --- start game loop    
            while True:
                #   handle events
                if self.handle_events() == QUIT_GAME:
                    manual_quit = True
                    break
                
                #   update sprites
                self.update()
                
                # check for snake collision
                if self.has_snake_collided() == QUIT_GAME:
                    break
                
                # check for snake growth
                self.handle_snake_growth()
                
                #   draw new game state
                self.draw()
                
                #   control speed
                self.clock.tick(FPS)
            
            if not self.looping or (self.looping and manual_quit):
                break
            
        #   end game
        pg.quit()
        sys.exit()
        
    def has_snake_collided(self):
        '''Function that helps detect game ending scenarios like:
            - snake hitting a wall
            - snake hitting itself
        Returns QUIT_GAME if either are true.'''
        
        # --- is snake hitting looping in on itself?
        hitting_self = self.snake.body[0] in self.snake.body[1:]
        
        # --- is snake out of bounds?
        snake_head = self.snake.body[0]
        #   horizontal check
        out_of_hor_bounds = (0 > snake_head[0]) or (WINDOW_WIDTH < snake_head[0])
        # vertical check
        out_of_ver_bounds = (0 > snake_head[1]) or (WINDOW_HEIGHT < snake_head[1])
        
        if any([hitting_self,out_of_hor_bounds,out_of_ver_bounds]):
            return QUIT_GAME
        else:
            return
    
    def handle_snake_growth(self):
        '''Function that handles snake getting the food.'''
        
        pass
        
    def update(self):
        '''Updates the game state.'''
        
        # snake
        self.snake.update()
        
        # food
        pass
    
    def draw(self):
        '''Draws new game state.'''
        
        # background
        self.screen.fill(self.board_color)
        
        # snake
        self.snake.draw(self.screen)
        
        # food
        pass
        
        # flip screen
        pg.display.flip()
    
    def handle_events(self):
        '''Handles keyboard and mouse inputs.'''
        
        for event in pg.event.get():
            # check for quit
            if event.type == pg.QUIT:
                return QUIT_GAME
            
            # check for arrow keys pressed
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_UP and self.snake.direction in [LEFT,RIGHT]:
                    self.snake.direction = UP
                if event.key == pg.K_DOWN and self.snake.direction in [LEFT,RIGHT]:
                    self.snake.direction = DOWN
                if event.key == pg.K_LEFT and self.snake.direction in [UP,DOWN]:
                    self.snake.direction = LEFT
                if event.key == pg.K_RIGHT and self.snake.direction in [UP,DOWN]:
                    self.snake.direction = RIGHT
                
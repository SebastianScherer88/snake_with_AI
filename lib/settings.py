# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 21:23:03 2018

@author: bettmensch
"""

# --- dimensions
WINDOW_WIDTH_PIXELS = 400
WINDOW_HEIGHT_PIXELS = 400

TILE_HEIGHT = TILE_WIDTH = 10
WINDOW_WIDTH = WINDOW_WIDTH_PIXELS / TILE_WIDTH
WINDOW_HEIGHT = WINDOW_HEIGHT_PIXELS / TILE_HEIGHT

# --- speed
FPS = 30

# --- visuals
BLUE = (0,0,255)
GREEN = (0,255,0)
RED = (255,0,0)
BLACK = (0,0,0)
WHITE = (255,255,255)

# --- constants
UP = 'UP'
RIGHT = 'RIGHT'
LEFT = 'LEFT'
DOWN = 'DOWN'
NOTHING = 'NOTHING'
QUIT_GAME = 'QUIT_GAME'
SNAKE_INIT_LENGTH = 10
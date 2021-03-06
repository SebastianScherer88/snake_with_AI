# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 21:23:03 2018

@author: bettmensch
"""

import numpy as np

# --- dimensions
WINDOW_WIDTH_PIXELS = 400
WINDOW_HEIGHT_PIXELS = 400

TILE_HEIGHT = TILE_WIDTH = 20
WINDOW_WIDTH = int(WINDOW_WIDTH_PIXELS / TILE_WIDTH)
WINDOW_HEIGHT = int(WINDOW_HEIGHT_PIXELS / TILE_HEIGHT)

# --- speed
FPS = 50

# --- visuals
BLUE = (0,0,255)
GREEN = (0,255,0)
RED = (255,0,0)
BLACK = (0,0,0)
WHITE = (255,255,255)

# --- game constants
UP = 'UP'
RIGHT = 'RIGHT'
LEFT = 'LEFT'
DOWN = 'DOWN'
NOTHING = 'NOTHING'
QUIT_GAME = 'QUIT_GAME'
SNAKE_INIT_LENGTH = 5
FONT_SIZE = 20
SCORE_OFF_X = WINDOW_WIDTH_PIXELS - TILE_WIDTH
SCORE_OFF_Y = TILE_HEIGHT

# --- AI constants
SNAKE_VALUE = 2
FOOD_VALUE = -2
DIRECTION_VALUE = 10
N_INPUT_FRAMES = 2
DIRECTION_TEMPLATE = np.array([DOWN,LEFT,UP,RIGHT]).reshape((1,-1))
#TURN_TEMPLATE = np.array([LEFT,UP,RIGHT]).reshape((1,-1))
TURN_TEMPLATE = np.array([LEFT,RIGHT]).reshape((1,-1))
N0 = 4
N1 = 8
#N2 = 3
N2 = 2
APPLY_AI_STEER = {(UP,UP):UP, # (turn_direction, current_direction) -> new_direction
                  (UP,LEFT):LEFT,
                  (UP,RIGHT):RIGHT,
                  (UP,DOWN):DOWN,
                  (LEFT,UP):LEFT,
                  (LEFT,LEFT):DOWN,
                  (LEFT,RIGHT):UP,
                  (LEFT,DOWN):RIGHT,
                  (RIGHT,UP):RIGHT,
                  (RIGHT,LEFT):UP,
                  (RIGHT,RIGHT):DOWN,
                  (RIGHT,DOWN):LEFT}

# --- GA constants
MAX_FRAMES = 1000
N_GENERATIONS = 60
N_POP = 30
MUTATION_RATE = 0.15

# --- PG constants
PG_LEARNING_RATE = 0.0001
PG_MOMENTUM = 0
PG_EPSILON = 0.1223523 # useless param but has to be specified - bug in optimizer class
PG_REG_PARAM = 0
PG_BATCH_SIZE = 1
MAX_FRAMES_PG = 40
N_EPISODES = 50000
POLICY_REWARD = 1
POLICY_REGRET = 1
P_EXPLORATION = 0.95
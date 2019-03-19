# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 17:11:04 2018

@author: bettmensch
"""

#----------------------------------------------------------------------
# [0] Imports and dependencies
#----------------------------------------------------------------------

from classes import Snake_With_AI
from classes import flat_input_generator
from classes import ai_from_ffnetwork
import numpy as np
from functools import partial
from diy_deep_learning_library import FFNetwork
from settings import *

#----------------------------------------------------------------------
# [1] Build snake piloting network
#----------------------------------------------------------------------

input_size = 4#WINDOW_HEIGHT * WINDOW_WIDTH + 4

neural_net = FFNetwork()

# bogus data sample; needed to fixate net
X_sample = np.random.normal(size=(10,input_size))

n1 = 100
n2 = 2

neural_net.addFCLayer(n1,activation='tanh')
neural_net.addFCLayer(n2,activation='softmax')

neural_net.fixateNetwork(X_sample)
neural_net.trained = True


# manually initialize ordered class array for classification model using turn template
neural_net.oneHotY(TURN_TEMPLATE)

print(neural_net)

#----------------------------------------------------------------------
# [2] Build both AI functions needed for AI snake simulation
#----------------------------------------------------------------------

# input generator
neural_input_generator = flat_input_generator

# ai steer
neural_ai = partial(ai_from_ffnetwork,neural_net)

#----------------------------------------------------------------------
# [3] Start snake game with AI
#----------------------------------------------------------------------

# --- set simulation params
#    max frames
max_frames = 500
#   fps
fps = 10
#   length of game state history needed
len_history = 1

# --- start AI simulation
snake_with_ai = Snake_With_AI(fps = fps,
                              looping = True,
                              use_ai = True,
                              max_frames = max_frames,
                              ai = neural_ai,
                              ai_input_generator = neural_input_generator,
                              len_history = len_history)

snake_with_ai.start()
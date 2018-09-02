# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 22:20:22 2018

@author: bettmensch
"""

#--------------------------------------------------------------
# [0] Imports & dependencies
#--------------------------------------------------------------

from functools import partial
from settings import *
from classes import build_policy_gradient_episode_generator
from classes import Snake_With_AI
from diy_deep_learning_library import  PG
from matplotlib import pyplot as plt
import numpy as np

#--------------------------------------------------------------
# [1] Build snake simulator tools
#--------------------------------------------------------------

neural_net, episode_generator = build_policy_gradient_episode_generator()

#--------------------------------------------------------------
# [2] Set policy gradient training params
#--------------------------------------------------------------



#--------------------------------------------------------------
# [2] Creat policy gradient network training wrapper and train AI
#--------------------------------------------------------------

pg_coach = PG(neural_net)

pg_coach.train_network(episode_generator,
                      n_episodes = N_EPISODES,
                      learning_rate = PG_LEARNING_RATE,
                      episode_batch_size = PG_BATCH_SIZE,
                      verbose = False)

MAX_FRAMES_PG = 40
N_EPISODES = 1000
POLICY_REWARD = 1
POLICY_DETENTION = -1
P_EXPLORATION = 0.3
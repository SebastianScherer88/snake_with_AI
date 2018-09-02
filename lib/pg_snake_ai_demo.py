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
from classes import ai_from_ffnetwork
from classes import flat_input_generator
from classes import Snake_With_AI
from diy_deep_learning_library import  PG
from matplotlib import pyplot as plt
import numpy as np

#--------------------------------------------------------------
# [1] Build snake simulator tools
#--------------------------------------------------------------

neural_net, episode_generator = build_policy_gradient_episode_generator()

neural_net.initialize_layer_optimizers('sgd',
                                       eta=PG_LEARNING_RATE,
                                       gamma=PG_MOMENTUM,
                                       epsilon=PG_EPSILON,
                                       lamda=PG_REG_PARAM,
                                       batchSize=PG_BATCH_SIZE)

#--------------------------------------------------------------
# [2] Creat policy gradient network training wrapper and train AI
#--------------------------------------------------------------

pg_coach = PG(neural_net)

pg_coach.train_network(episode_generator,
                      n_episodes = N_EPISODES,
                      learning_rate = PG_LEARNING_RATE,
                      episode_batch_size = PG_BATCH_SIZE,
                      verbose = True,
                      reward=POLICY_REWARD,
                      regret = POLICY_REGRET)

#--------------------------------------------------------------
# [3] Watch AI in action
#--------------------------------------------------------------

neural_ai = partial(ai_from_ffnetwork,
                    neural_net)

neural_input_generator = flat_input_generator

Snake_With_AI(fps = 15,
                 looping = True,
                 use_ai = True,
                 max_frames = 1000,
                 ai = neural_ai,
                 ai_input_generator = neural_input_generator,
                 len_history = N_INPUT_FRAMES,
                 visuals = True,
                 speed_limit = True).start()
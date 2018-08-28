# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 20:27:33 2018

@author: bettmensch
"""

#--------------------------------------------------------------
# [0] Imports & dependencies
#--------------------------------------------------------------

from functools import partial
from settings import *
from classes import build_ai_simulation_tools, build_ai_simulation_cost_function
from diy_deep_learning_library import GeneWeightTranslator, GA
from matplotlib import pyplot as plt
import numpy as np

#--------------------------------------------------------------
# [1] Build snake simulator tools
#--------------------------------------------------------------

neural_net, neural_input_generator, neural_ai = build_ai_simulation_tools()

#--------------------------------------------------------------
# [2] Build genetic algorithm inputs
#--------------------------------------------------------------

# --- initializer
gene_weight_trans = GeneWeightTranslator(neural_net)
gene_initializer = gene_weight_trans.initialize_genes

# --- cost function
gene_score_function = partial(build_ai_simulation_cost_function,
                              genes_to_weight_translator = gene_weight_trans,
                              neural_net = neural_net,
                              neural_input_generator = neural_input_generator,
                              neural_ai = neural_ai)

#--------------------------------------------------------------
# [3] Start up snake AI genetic algorithm
#--------------------------------------------------------------

# initialize algorithm object
snake_ga = GA(gene_weight_trans.dna_seq_len)

# start evolution
snake_ga.evolve(cost_function = gene_score_function,
               max_gens = 10,
               n_pop = N_POP,
               mutation_rate = MUTATION_RATE)

#--------------------------------------------------------------
# [4] Visualize results
#--------------------------------------------------------------

# --- plot score history
score_hist = snake_ga.population_history.groupby('n_gen')
#   means
score_means = score_hist.mean()['score']
plt.plot(score_means, label = 'Mean over generation', color = 'g')
#   stdev envelope
score_std = score_hist.std()['score']
upper_bound = score_means + score_std
lower_bound = score_means - score_std
plt.plot(upper_bound, label = 'Mean + Std',color = 'b')
plt.plot(lower_bound, label = 'Mean - Std',color = 'b')
#   max
score_max = score_hist.max()['score']
plt.plot(score_max, label = 'Max over generation', color = 'r')
# show plot
plt.show()

# --- watch best gene in action

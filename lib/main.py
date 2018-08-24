# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 22:32:59 2018

@author: bettmensch
"""

from classes import Snake
from classes import Snake_With_AI

def main():
    # get new game object
    new_game = Snake_With_AI()
    
    # start game
    new_game.start()
    
if __name__ == '__main__':
    main()
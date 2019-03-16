# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 22:32:59 2018

@author: bettmensch
"""

from classes import Snake
from classes import Snake_With_AI

def main():
    # get new game object
    new_game = Snake_With_AI(fps = 15,
                             use_ai = False,
                             looping = False)
    
    # start game
    total_score = new_game.start()
    
    # print total score
    print("Total score:", total_score)
    
if __name__ == '__main__':
    main()
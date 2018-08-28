# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 21:27:13 2018

@author: bettmensch
"""

#Content
# [0] Imports & dependencies
# [1] Util function: draw_tile
# [2] Object class: Snake (snake game sprite)
# [3] Object class: Snake_With_AI (game/simulation)
# [4] Util function: input_generator
# [5] Util function: snake_ai

#-------------------
# [0] Imports & dependencies
#-------------------

import pygame as pg
import numpy as np
import sys
import random
from settings import *
from functools import partial
from diy_deep_learning_library import FFNetwork


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

#-------------------
# [1] Object class: Snake (snake game sprite)
#-------------------


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
        
        # last frames tail of snake body - needed for eating logic
        self.previous_tail = self.body[-1]
        
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
        
        #   update previous frame's tail
        self.previous_tail = self.body[-1]
        
        #   delete old tail
        self.body.pop(-1)
        
    def draw(self,
             screen):
        '''Gets called once every frame. Draws the snake to the screen.'''
        
        for snake_tile_position in self.body:
            draw_tile(screen,
                      snake_tile_position,
                      self.color)
            
#-------------------
# [2] Object class: Snake_With_AI (game/simulation)
#-------------------
            
class Snake_With_AI(object):
    '''Game class. Represent a simulation of a game of snake.
    Allows for playing normally, but also allows plugging in a (neural
    network based) AI of a certain format to run simulations. In particular,
    can be used as a building block for a GA or RL based AI training routine.'''
    
    def __init__(self,
                 fps = FPS,
                 looping = True,
                 use_ai = True,
                 max_frames = MAX_FRAMES,
                 ai = None,
                 ai_input_generator = None,
                 len_history = 1,
                 visuals = True,
                 speed_limit = True):
        
        # --- essential params
        self.fps = fps
        self.board_color = WHITE
        self.food_color = RED
        self.text_color = BLACK
        self.clock = pg.time.Clock()
        self.screen = None
        self.snake = None
        self.food = None
        self.score = None
        self.total_score = None
        self.looping = looping
        self.using_ai = False
        self.visuals = visuals
        self.speed_limit = speed_limit
        
        # --- training routine mode params & methods
        
        # make sure all tools are specified to make AI training routine feasible
        if use_ai:
            assert(all([max_frames != None,
                        ai != None,
                        ai_input_generator != None]))
            # switch flick
            self.using_ai = True
            # training routing tracking params
            self.looping = True # override potential False; no training routinge makes sense without repetitions
            self.n_frames_passed = None
            self.max_frames = max_frames # Training routine parameter; sensible unit to guarantee constant
            self.len_history = len_history
            self.state_history = None
            # attach piloting logic encoded in specified ai model
            self.ai = ai
            # attach function generating inputs for specified ai model
            self.ai_input_generator = ai_input_generator
        
        # --- set up pygame window
        #   start up pygame
        pg.init()
        
        if self.visuals:
            # intialize graphic window
            self.screen = pg.display.set_mode((WINDOW_WIDTH_PIXELS,WINDOW_HEIGHT_PIXELS))
            #   set caption
            pg.display.set_caption("SNAKE WITH AI")
            # fill background
            self.screen.fill(self.board_color)
            
            self.font = pg.font.Font('freesansbold.ttf',FONT_SIZE)
        
    def start(self):
        '''Called to start a new game.'''
        
        # initialize total score
        self.total_score = 0
        
        # initialize frames passed
        self.n_frames_passed = 0
        
        # if game is looping, this loop causes infinite games
        while True:
            # --- draw game sprites
            #   snake
            self.snake = Snake()
            if self.visuals:
                self.snake.draw(self.screen)
        
            # first food
            self.food = self.get_new_food_position()
        
            # --- stop flags
            #    manual stop control flag
            manual_quit = False
            #    simluation stop flag if appropriate
            simulation_quit = False
            
            # reset score
            self.score = 0
            
            # initialize state history if needed
            if self.using_ai:
                self.initialize_state_history()
        
            # --- start game loop    
            while True:
                # if in AI mode, check if max frame number has been reached
                if self.using_ai:
                    if self.n_frames_passed == self.max_frames:
                        simulation_quit = True
                        break
                    
                # record game state and add to history of recent n game states
                if self.using_ai:
                    self.record_current_state()
                
                # handle events - snake pilot commands are produced & processed here; also, manualy closing the pygame window
                if self.handle_events() == QUIT_GAME:
                    manual_quit = True
                    break
                
                # update sprites - snake position is updated here
                self.update()
                
                # check for snake collision
                if self.has_snake_collided() == QUIT_GAME:
                    self.total_score = max(0,self.total_score - 1)
                    break
                
                #   draw new game state
                if self.visuals:
                    self.draw()
                
                #   control speed
                if self.speed_limit:
                    self.clock.tick(self.fps)
            
            if not self.using_ai:
                if not self.looping or (self.looping and manual_quit):
                    break
            elif self.using_ai:
                if simulation_quit or manual_quit:
                    break
            
        #   end (looping) game / simulation
        pg.quit()
        
        return self.total_score
        
    def handle_snake_food(self):
        '''Function that handles distribution of new food and the snake colliding
        with existing food.'''
        
        # --- check if snake has reached food
        snake_head = self.snake.body[0]
        previous_snake_tail = self.snake.previous_tail
        
        if snake_head == self.food:
            # update scores
            self.score += 1
            self.total_score += 1
            
            # re-attach previous frame's tail to make snake grow
            self.snake.body += [previous_snake_tail]
            
            # --- place new food object
            self.food = self.get_new_food_position()
        
    def get_new_food_position(self):
        '''Randomly creats a new location for food.'''
        
        # --- get random location on board
        while True:
            #   get random tile location on board
            new_food = [random.randint(0,WINDOW_WIDTH-1),random.randint(0,WINDOW_HEIGHT-1)]
            #   if not on snake body, use this valid location
            if not (new_food in self.snake.body):
                break

        return new_food
        
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
        
    def update(self):
        '''Updates the game state.'''
        
        # update frame counter if appropriate
        if self.using_ai:
            self.n_frames_passed += 1
        
        # snake
        self.snake.update()
        
        # food
        self.handle_snake_food()
    
    def draw(self):
        '''Draws new game state.'''
        
        # background
        self.screen.fill(self.board_color)
        
        # snake
        self.snake.draw(self.screen)
        
        # food
        draw_tile(self.screen,
                  self.food,
                  self.food_color)
        
        # --- score
        #   get text
        score_message = "Score: " + str(self.score)
        #   get text surface
        score_surf = self.font.render(score_message,
                                      False,
                                      self.text_color)
        #   position text surface
        score_rect = score_surf.get_rect()
        score_rect.right = SCORE_OFF_X
        score_rect.top = SCORE_OFF_Y
        
        self.screen.blit(score_surf,
                         score_rect)
        
        # --- total score
        if self.using_ai:
            #   get text
            score_message = "Total gene's score: " + str(self.total_score)
            #   get text surface
            score_surf = self.font.render(score_message,
                                          False,
                                          self.text_color)
            #   position text surface
            score_rect = score_surf.get_rect()
            score_rect.left = TILE_WIDTH
            score_rect.top = SCORE_OFF_Y
        
        self.screen.blit(score_surf,
                         score_rect)
        
        # flip screen
        pg.display.flip()
    
    def handle_events(self):
        '''Handles keyboard and mouse inputs.'''
        
        for event in pg.event.get():
            # check for quit
            if event.type == pg.QUIT:
                return QUIT_GAME
            
            # check for arrow keys pressed
            if event.type == pg.KEYDOWN and not self.using_ai:
                if event.key == pg.K_UP and self.snake.direction in [LEFT,RIGHT]:
                    self.snake.direction = UP
                if event.key == pg.K_DOWN and self.snake.direction in [LEFT,RIGHT]:
                    self.snake.direction = DOWN
                if event.key == pg.K_LEFT and self.snake.direction in [UP,DOWN]:
                    self.snake.direction = LEFT
                if event.key == pg.K_RIGHT and self.snake.direction in [UP,DOWN]:
                    self.snake.direction = RIGHT
        
        # if in AI mode, get AI steering prediction
        if self.using_ai:
            # generate input for AI from raw game state history
            ai_input = self.ai_input_generator(self.state_history)
            # get AI steer
            ai_turn = self.ai(ai_input)
            # apply AI steer
            current_direction = self.snake.direction
            self.snake.direction = APPLY_AI_STEER[(ai_turn,current_direction)]
                    
    def initialize_state_history(self):
        '''Util function that initializes the game's raw state history by padding
        it with self.history_len empty states. This is needed so the AI can make
        (generic) decisions at the beginning of each game when there are no past
        game states.'''
        
        self.state_history = [{'snake_pos':None,
                               'snake_dir':None,
                               'food_pos':None,
                               'score':None}] * self.len_history
                    
    def record_current_state(self):
        '''Util function that saves the game state of the current frame and appends
        it to the current game's state history. Needed to create raw data which is
        then picked up by the ai input generator.'''
        
        current_state = {'snake_pos':self.snake.body,
                         'snake_dir':self.snake.direction,
                         'food_pos':self.food,
                         'score':self.score}
        
        # update and cut down to size
        self.state_history.append(current_state)
        self.state_history = self.state_history[-self.len_history:]
        
#-------------------
# [4] Util function: input_generator
#-------------------
        
def flat_input_generator(raw_history,
                         food_value=FOOD_VALUE,
                         snake_value=SNAKE_VALUE):
    '''Util function that takes the most recent raw snake game state and converts
    it into a vector that quantifies 
        - current snake position (in tile coordinate system)
        - current snake direction
        - current food position (in tile coordinate system)'''
    
    # get most recent raw state
    raw_state = raw_history[-1]
    
    # --- build quantified board state
    #   initialize tiled coordinate system
    board_state = np.zeros((WINDOW_WIDTH,WINDOW_HEIGHT))
    #   add food
    #print("Food pos:",raw_state['food_pos'])
    food_tile_x, food_tile_y = raw_state['food_pos']
    
    board_state[food_tile_x,food_tile_y] += FOOD_VALUE
    #    add snake
    for (snake_tile_x,snake_tile_y) in raw_state['snake_pos']:
        # verify tile is on board
        if (snake_tile_x in range(WINDOW_WIDTH)) and (snake_tile_y in range(WINDOW_HEIGHT)):
            board_state[snake_tile_x,snake_tile_y] += SNAKE_VALUE
    # flatten board to vector
    flat_board_state = board_state.reshape((1,-1))
    
    # --- build snake direction state
    snake_dir = np.array([raw_state['snake_dir']] * 4).reshape((1,-1))
    flat_direction_state = (DIRECTION_TEMPLATE == snake_dir) * DIRECTION_VALUE
    
    # --- combine to total state
    state = np.concatenate([flat_board_state,
                            flat_direction_state],
                            axis=1)
    
    return state

        
#-------------------
# [5] Util function: snake_ai
#-------------------

def ai_from_ffnetwork(ffnetwork,
             input_state):
    '''Util wrapper around specified FFNetwork that takes an input array of shape (1,d_input)
    and return one of directional constants UP, DOWN, RIGHT or LEFT.'''
    
    # verify that network is ready
    assert(ffnetwork.finalState)
    
    # get prediction array
    prediction = ffnetwork.predict(input_state)
    # get direction
    direction = TURN_TEMPLATE[0,prediction][0,0]
    
    return direction

#--------------------------------------------------------------
# [6] Util function:  build AI simulation dependencies
#--------------------------------------------------------------
    
def build_ai_simulation_tools():
    '''Util function that produces tools and inputs needed for the AI snake 
    simulation.'''

    # --- build neural net
    input_size = WINDOW_HEIGHT * WINDOW_WIDTH + np.prod((DIRECTION_TEMPLATE.shape))
    
    neural_net = FFNetwork()
    
    # bogus data sample; needed to fixate net
    X_sample = np.random.normal(size=(10,input_size))
    
    neural_net.addFCLayer(N1,activation='tanh')
    neural_net.addFCLayer(N2,activation='softmax')
    
    # fixate and flick training switch
    neural_net.fixateNetwork(X_sample)
    neural_net.trained = True

    print(neural_net)

    # --- input generator
    neural_input_generator = flat_input_generator
    
    # --- neural net based snake ai
    neural_ai = partial(ai_from_ffnetwork,neural_net)
    
    return neural_net, neural_input_generator, neural_ai

#--------------------------------------------------------------
# [7] Util function:  build genetic algorithm cost function for AI snake task
#--------------------------------------------------------------
    
def build_ai_simulation_cost_function(gene,
                                      genes_to_weight_translator,
                                      neural_net,
                                      neural_input_generator,
                                      neural_ai):
    '''Util function to which we will apply the functools::partial function to 
    obtain a cost function suitabel for the AI snake genetic algorithm evolution.'''
    
    # verify that translator was built on the network specified as argument
    assert(genes_to_weight_translator.ffnetwork == neural_net)
    
    #print('Gene shape:',gene.shape)
    #print('Gene shape:',gene[0].shape)
    
    # --- genes -----> cost function value pipeline
    #   convert genes to weights
    weights = genes_to_weight_translator.gene_to_weights(gene)
    
    #   set neural network weights
    genes_to_weight_translator.set_current_weights(weights)# the weights of 'neural_net' are being set here!
    
    #   set up simulation
    gene_score = Snake_With_AI(ai=neural_ai,
                               ai_input_generator=neural_input_generator,
                               visuals=False,
                               speed_limit=False).start()
    
    return float(gene_score)
    
    
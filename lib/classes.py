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
                 len_history = N_INPUT_FRAMES,
                 visuals = True,
                 speed_limit = True,
                 using_ga = True,
                 using_pg = False):
        
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
            # only one of these two methods allowed
            assert(using_ga != using_pg)
            # flick switch
            self.using_ai = True
            self.using_ga = using_ga
            self.using_pg = using_pg
            # training routing tracking params
            self.looping = True # override potential False; no training routinge makes sense without repetitions
            self.n_frames_passed = None
            self.max_frames = max_frames # Training routine parameter; sensible unit to guarantee constant
            self.len_history = len_history
            self.state_history = None
            self.action_history = None
            self.i_episode = 0
            self.failed_episode = True
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
                    pg.display.flip()
                
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
    
    def run_pg_episode(self,
                       p_exploration=P_EXPLORATION):
        '''Starts a new online learning epsiode that generates and returns a batch
        for policy gradient based AI training. Either starts new game (if previous 
        episode ended with failure) or picks up where last episode ended (if previous
        epsiode ended with success.)'''
        
        # ensure game is in correct mode
        assert(self.using_pg)
        assert(self.using_ai)
                
        # initialize empty padded state and action history for this episode
        self.initialize_state_history()
        self.initialize_action_history()
        self.n_frames_passed = 0
        
        if self.i_episode == 0:
            self.score = 0
            self.total_score = 0
                                                                       
        # update episode counter
        self.i_episode += 1
                                                                       
        # --- check outcome of previous episode
        if self.failed_episode:
            # --- initialize game objects
            #   snake
            self.snake = Snake()        
            #   first food
            self.food = self.get_new_food_position()
        else:
            # reuse last epsiode's snake and food; still attached to game
            pass
        
        # --- (re-)start game loop
        while True:
            # check if max frame number has been reached
            if self.n_frames_passed == self.max_frames:
                # if no food was found but snake hasnt collided yet, discourage AI form doing this in future
                self.failed_episode = True
                rl_coeff = -1
                if self.visuals:
                    self.show_pg_epsiode_commercial_break(self.failed_episode)
                break
                
            # --- record game state and action and add to history of episode
            #   snake pilot commands are produced & processed here
            self.record_current_state()
            action = self.apply_pg_ai_steer(p_exploration)
            self.record_current_action(action)

            # update sprites - snake position is updated here
            food_found = self.update()
            if food_found:
                self.failed_epsiode = False
                rl_coeff = 1
                if self.visuals:
                    self.show_pg_epsiode_commercial_break(self.failed_episode)
                break
            
            # check for snake collision
            if self.has_snake_collided() == QUIT_GAME:
                self.failed_episode = True
                rl_coeff = -1
                if self.visuals:
                    self.show_pg_epsiode_commercial_break(self.failed_episode)
                break
            
            #   draw new game state
            if self.visuals:
                self.draw()
            
            #   control speed
            if self.speed_limit:
                self.clock.tick(self.fps)
                
        # return the rl coefficient and the processed states and actions
        processed_states = self.process_state_history()
        processed_actions = self.process_action_history()
        
        return processed_states, processed_actions, rl_coeff
                
    def process_state_history(self):
        '''Util function that converts all the raw states currently held in 
        self.state_history to an (n_episode_frames,d_input) array of processed inputs.'''
        
        # get processed states (via slices of state history to ensure backwards compat. with GA input processor)
        hist = self.state_history
        processed_states = [self.ai_input_generator(hist[:i_end]) for i_end in range(1,len(hist)+1)]
        X_states = np.array(processed_states).reshape((len(processed_states),-1))
        
        return X_states
    
    def process_action_history(self):
        '''Util function that converts all the actions currently held in 
        self.action_history to an (n_episode_frames,?) array of processed inputs.'''
        
        return np.array(self.action_history).reshape((-1,1))
                
    def show_pg_epsiode_commercial_break(self,
                                         failed):
        '''Util function that prints "Updating AI..." to the pygame screen at 
        appropriate times during policy gradient AI training routine'''
        
        # get update message
        message = "Failed! " * failed + "Success! " * (1 - failed) + "Updating AI..."
        
        #   get text surface
        billboard_surf = self.font.render(message,
                                      False,
                                      RED)
        
        #   position text on pygame window
        billboard_rect = billboard_surf.get_rect()
        billboard_rect.center = (int(WINDOW_WIDTH_PIXELS / 2),int(WINDOW_HEIGHT_PIXELS / 2))
        
        #   blit message and display
        self.screen.blit(billboard_surf,billboard_rect)
        pg.display.flip()
            
    def apply_pg_ai_steer(self,p_explore):
        '''Util function for AI steering during polcy gradient AI training routine.'''

        # exploration or exploitation?
        lets_go_exploring = np.random.uniform() < p_explore
        
        # --- explore
        if lets_go_exploring:
            # get random choice
            turn = np.random.choice(TURN_TEMPLATE.reshape(-1))
            
        # --- exploit
        elif not lets_go_exploring:
            # get AI choice based on state
            # generate input for AI from raw game state history
            ai_input = self.ai_input_generator(self.state_history)
            #print("ai_input:",ai_input)
            # get AI steer
            turn = ai_turn = self.ai(ai_input)
            
        # apply random/AI turn
        current_direction = self.snake.direction
        self.snake.direction = APPLY_AI_STEER[(turn,current_direction)]
        
        #print(turn)
        
        return turn
        
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
        food_found = self.handle_snake_food()
        
        return food_found
    
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
        if self.using_ai and self.using_ga:
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
        
        # --- policy gradient episode
        if self.using_ai and self.using_pg:
            #   get text
            episode_message = "Epsiode #: " + str(self.i_episode)
            #   get text surface
            episode_surf = self.font.render(episode_message,
                                          False,
                                          self.text_color)
            #   position text surface
            episode_rect = episode_surf.get_rect()
            episode_rect.left = TILE_WIDTH
            episode_rect.top = SCORE_OFF_Y
        
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
        
        #self.state_history = [{'snake_pos':[0,0],
        #                       'snake_dir':LEFT,
        #                       'food_pos':[0,0],
        #                       'score':0}] * self.len_history
        self.state_history = []
    
    def initialize_action_history(self):
        '''Util function that initializes the game's raw action history by padding
        it with self.history_len empty states. This is needed so the AI can make
        (generic) decisions at the beginning of each game when there are no past
        game actions.'''
        
        #self.action_history = [LEFT,] * self.len_history
        self.action_history = []
                    
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
        
    def record_current_action(self,
                              current_action):
        '''Util function that saves the action of the current frame and appends it to 
        the the current game's action history. Needed to create raw data which is 
        then picked up by the ai target generator.'''
        
        self.action_history.append(current_action)
        self.action_history = self.action_history[-self.len_history:]
        
        
        
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
    raw_state_current = raw_history[-1]
    
    # --- build quantified board state
    #   get relative coordinates to food for current frame and normalize
    rel_food_pos_current = (np.array(raw_state_current['food_pos']) - np.array(raw_state_current['snake_pos'][0])).reshape((1,-1)) \
    / np.array([WINDOW_WIDTH_PIXELS,WINDOW_HEIGHT_PIXELS])
    # get direction coordinates
    dir_current_raw = raw_state_current['snake_dir']
    if dir_current_raw == UP:
        dir_current = [0,-1]
    elif dir_current_raw == DOWN:
        dir_current = [0,1]
    elif dir_current_raw == LEFT:
        dir_current = [-1,0]
    elif dir_current_raw == RIGHT:
        dir_current = [1,0]
    dir_current = 0.5 * np.array(dir_current).reshape(1,-1)
    # combine into one input vector
    state = np.concatenate([rel_food_pos_current,
                            dir_current],
                            axis=1)
    
    return state

        
#-------------------
# [5] Util function: snake_ai
#-------------------

def ai_from_ffnetwork(ffnetwork,
             input_state):
    '''Util wrapper around specified FFNetwork that takes an input array of shape (1,d_input)
    and return one of directional constants UP, DOWN, RIGHT or LEFT.'''
    
    #print(ffnetwork.classes_ordered)
    
    # verify that network is ready
    assert(ffnetwork.finalState)
    
    # get prediction array
    prediction = ffnetwork.predict(input_state)
    #print("Classes ordered of ai neual net:",ffnetwork.classes_ordered)
    
    #print("prediction:",prediction)
    # get direction
    direction = prediction[0][0]
    
    return direction

#--------------------------------------------------------------
# [6] Util function:  build AI simulation dependencies
#--------------------------------------------------------------
    
def build_ai_simulation_tools():
    '''Util function that produces tools and inputs needed for the AI snake 
    simulation.'''

    # --- build neural net
    input_size = N0
    
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
    
#--------------------------------------------------------------
# [8] Util function:  build policy gradient episode generator function
#--------------------------------------------------------------
    
def build_policy_gradient_episode_generator(visuals=False,
                                            speed_limit=False):
    '''Util function that creates processed AI input data for one episode
    by running a snake simulation up until the first non-trivial reward and 
    recording all (state,action) pairs in the process.
    
    Returns a handle to the neural net used as AI as well as the acutal
    generating function.'''
    
    # build tools for snake simluation
    neural_net, neural_input_generator, neural_ai = build_ai_simulation_tools()
    
    # create snake simulation in policy gradient mode with above tools
    snake_sim = Snake_With_AI(max_frames = MAX_FRAMES_PG,
                              ai = neural_ai,
                              ai_input_generator = neural_input_generator,
                              len_history = MAX_FRAMES_PG,
                              using_ga = False,
                              using_pg = True,
                              visuals=visuals,
                              speed_limit=speed_limit)
    
    # yoink the run episode method for outside use
    run_snake_episode = partial(snake_sim.run_pg_episode)
    
    return neural_net, run_snake_episode
    
    
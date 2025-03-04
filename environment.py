import pygame
from collections import deque
from movement import Direction, Square, handle_movement
import numpy as np


class SnakeEnvironment:
    def __init__(self, config): # config must be a dictionary containing 'GRID_WIDTH' and 'GRID_HEIGHT' entries
        self.snake1 = deque()
        self.snake2 = deque()
        self.tomatoes = []
        self.grid = [[Square.EMPTY] * config['GRID_WIDTH'] for _ in range(config['GRID_HEIGHT'])]
        self.head1_dir = Direction.RIGHT  # FIXME: why does this break if not initialized here?
        self.head2_dir = Direction.LEFT
        self.snake1_length = 0
        self.snake2_length = 0

        self.config = config

        self.actions = {0:Direction.LEFT, 1:Direction.UP, 2:Direction.RIGHT, 3:Direction.DOWN}
        self.move_counter = 0
        self.truncate_limit = 50000

        num_cells = config['GRID_WIDTH'] * config['GRID_HEIGHT']
        self.snake_arr_size = num_cells // 2 + 1

        # Hard coded - starts with snake lengths of 2
        self.reset(2, 2)

    def reset(self, snake1_length=2, snake2_length=2):
        snake1, snake2 = deque(), deque()
        grid = [[Square.EMPTY] * self.config['GRID_WIDTH'] for _ in range(self.config['GRID_HEIGHT'])] # coordinates are (y,x)
        
        for i in range(0, snake1_length):
            snake1.append((1,i))
            grid[1][i] = Square.PLAYER1
        
        for j in range(19, 19 - snake2_length, -1):
            snake2.appendleft((17, j))
            grid[17][j] = Square.PLAYER2

        head1_dir = Direction.RIGHT
        head2_dir = Direction.LEFT

        tomatoes = []
        tomatoes.append((2, 2))
        tomatoes.append((16, 16))
        tomatoes.append((10, 10))
        tomatoes.append((18, 2))
        tomatoes.append((2, 16))
        self.tomatoes = tomatoes
        for coord in tomatoes:
            row = coord[0]
            col = coord[1]
            grid[row][col] = Square.TOMATO

        self.grid, self.snake1, self.head1_dir, self.snake1_length, self.snake2, self.head2, self.snake2_length = grid, snake1, head1_dir, snake1_length, snake2, head2_dir, snake2_length
        return self.get_network_state()
    
    def step(self, input1, input2, update_state=True):
        old_snake1_length = self.snake1_length
        grid, snake1, head1_dir, snake1_length, snake2, head2_dir, snake2_length, tomatoes = handle_movement(game_state=self.get_game_state(), input1=input1, input2=input2, config=self.config)

        if update_state:
            self.grid, self.snake1, self.head1_dir, self.snake1_length, self.snake2, self.head2_dir, self.snake2_length, self.tomatoes = grid, snake1, head1_dir, snake1_length, snake2, head2_dir, snake2_length, tomatoes

        terminated = False
        if snake1 is None and snake2 is None:
            terminated = True
        if (snake1 is None and snake1_length < snake2_length) or (snake2 is None and snake1_length > snake2_length):
            terminated = True
        
        self.move_counter += 1
        truncated = self.move_counter >= self.truncate_limit

        # FIXME: reward handled only for snake 1 right now 
        reward1, reward2 = 0, 0
        if terminated:
            reward1 += 100 if snake1_length > snake2_length else -100
        
        # if snake2 and not self.snake2:  # Reward for killing snake 2
        #     reward += 50

        if snake1_length > old_snake1_length:  # New length is greater than old length, meaning snake ate a tomato
            reward1 += 1

        return (self.get_network_state(), reward1, reward2, terminated, truncated)
        
    
    def get_game_state(self):
        return (self.grid, self.snake1, self.head1_dir, self.snake1_length, self.snake2, self.head2_dir, self.snake2_length, self.tomatoes)
    
    def get_network_state(self, is_snake1=True):  # FIXME: Don't call np.array(None) or it returns a scalar array with None as the only value
        # FIXME: We need to change how we represent a snake being dead
        snake1 = np.array(self.snake1)
        snake1_padding = self.snake_arr_size - snake1.shape[0]
        if snake1_padding > 0:
            snake1 = np.pad(snake1, (0, snake1_padding), mode='constant', constant_values=0)

        snake2 = np.array(self.snake2)
        snake2_padding = self.snake_arr_size - snake2.shape[0]
        if snake2_padding > 0:
            snake2 = np.pad(snake2, (0, snake2_padding), mode='constant', constant_values=0)

        tomatoes = np.array(self.tomatoes)
        if is_snake1:
            return np.concatenate((snake1.flatten(), snake2.flatten(), tomatoes.flatten()))
        else:
            return np.concatenate((snake2.flatten(), snake1.flatten(), tomatoes.flatten()))

    
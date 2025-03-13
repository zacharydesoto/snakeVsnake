import pygame
from collections import deque
from movement import *
import numpy as np
import torch
import pickle
from utils import *

class SnakeEnvironment:
    def __init__(self, config):
        self.snake1 = None
        self.snake2 = None
        self.move_counter = None
        self.grid =  None
        self.tomatoes = None

        self.config = config
        self.truncate_limit = 5000
        num_cells = config['GRID_WIDTH'] * config['GRID_HEIGHT']
        self.snake_arr_size = num_cells // 2 + 1
        self.actions = [0, 1, 2]  # Left, straight, right
        self.int_to_action = {0: Direction.LEFT, 1: Direction.UP, 2: Direction.RIGHT, 3: Direction.DOWN}
        self.action_to_int = {Direction.LEFT: 0, Direction.UP: 1, Direction.RIGHT: 2, Direction.DOWN: 3}
        
        self.reset()
    
    def reset(self):
        self.grid = [[Square.EMPTY] * self.config['GRID_WIDTH'] for _ in range(self.config['GRID_HEIGHT'])]

        path1 = deque()
        path1.appendleft((1, 1))
        path1.appendleft((1, 2))
        path2 = deque()
        path2.appendleft((18, 19))
        path2.appendleft((18, 18))

        for snake1_square in path1:
            self.grid[snake1_square[0]][snake1_square[1]] = Square.PLAYER1
        for snake2_square in path2:
            self.grid[snake2_square[0]][snake2_square[1]] = Square.PLAYER2

        head_dir1 = Direction.RIGHT
        head_dir2 = Direction.LEFT

        snake_length1 = 2
        snake_length2 = 2

        self.tomatoes = [(10, 10), (1, 10), (19, 10), (5, 1), (5, 2)]
        for tomato in self.tomatoes:
            self.grid[tomato[0]][tomato[1]] = Square.TOMATO

        self.snake1 = SnakeData(path1, head_dir1, snake_length1, True)
        self.snake2 = SnakeData(path2, head_dir2, snake_length2, True)

        self.move_counter = 0
        self.truncate_limit = 5000


    def step(self, action1, action2):
        # Makes sure actions are Direction objects

        if type(action1) == int:
            current_dir = self.action_to_int[self.snake1.head_dir]
            new_dir = (current_dir + action1 - 1) % len(self.int_to_action)
            action1 = self.int_to_action[new_dir]
        if type(action2) == int:
            current_dir = self.action_to_int[self.snake2.head_dir]
            new_dir = (current_dir + action2  - 1) % len(self.int_to_action)
            action2 = self.int_to_action[new_dir]

        # Only updates the direction if the snake is turning
        if check_perpendicular_directions(self.snake1.head_dir, action1):
            self.snake1.head_dir = action1
        if check_perpendicular_directions(self.snake2.head_dir, action2):
            self.snake2.head_dir = action2

        # Checks if snakes went out of bounds or collided with something
        head1, self.snake1.alive = check_ob_collisions(self.grid, self.snake1.path, self.snake1.head_dir, self.snake1.alive, self.config)
        head2, self.snake2.alive = check_ob_collisions(self.grid, self.snake2.path, self.snake2.head_dir, self.snake2.alive, self.config)

        # Kills both snakes if they moved into the same square
        if (head1 is not None and head2 is not None) and head1 == head2:
            self.snake1.alive = False
            self.snake2.alive = False
        
        # Moves snakes, growing their length if they ate a tomato
        self.grid, self.snake1.path, snake_ate1 = update_snake(self.grid, self.snake1, Square.PLAYER1)
        self.grid, self.snake2.path, snake_ate2 = update_snake(self.grid, self.snake2, Square.PLAYER2)

        # Add new tomatos if any were eaten
        if snake_ate1:
            self.tomatoes.remove(head1)
            new_pos = add_tomato(self.grid, self.config)
            self.tomatoes.append(new_pos)
            self.snake1.length += 1
        if snake_ate2:
            self.tomatoes.remove(head2)
            new_pos = add_tomato(self.grid, self.config)
            self.tomatoes.append(new_pos)
            self.snake2.length += 1
        
        # Update game information
        terminated = (not self.snake1.alive) and (not self.snake2.alive)
        self.move_counter += 1
        truncated = self.move_counter >= self.truncate_limit
        if truncated:
            print('Game truncated.')

        # Calculate Reward
        reward1, reward2 = 0, 0
        if snake_ate1:
            reward1 += 10
        if snake_ate2:
            reward2 += 10
        if terminated:
            reward1 -= 10
            reward2 -= 10

        # Return state, reward, terminated, truncated
        return (reward1, reward2, terminated, truncated)
    
    def get_network_state(self, is_snake1=True):
        '''Returns current state of game in form (danger_left, danger_up, danger_right, danger_down, food_left, food_up, food_right, food_down)'''
        snake = self.snake1 if is_snake1 else self.snake2
        head = snake.path[0]

        # Handles relative positions by accounting for head direction by rotation functions
        if snake.head_dir == Direction.LEFT:
            abs_to_rel_offset = lambda tup: (tup[1], -tup[0])
            rel_to_abs_offset = lambda tup: (-tup[1], tup[0])
        elif snake.head_dir == Direction.UP:
            abs_to_rel_offset = lambda tup: (tup[0], tup[1])
            rel_to_abs_offset = lambda tup: (tup[0], tup[1])
        elif snake.head_dir == Direction.RIGHT:
            abs_to_rel_offset = lambda tup: (-tup[1], tup[0])
            rel_to_abs_offset = lambda tup: (tup[1], -tup[0])
        else:
            abs_to_rel_offset = lambda tup: (-tup[0], -tup[1])
            rel_to_abs_offset = lambda tup: (-tup[0], -tup[1])

        state = []
        danger_offsets = [(0, -1), (-1, 0), (0, 1)] # Left, straight, right
        danger_offsets.extend([(-1, -1), (-1, 1)]) # Straight-left, straight_right
        danger_offsets.extend([(-2, -1), (-2, 0), (-2, 1)]) # 2 spaces forward
        danger_offsets.extend([(0, -2), (0, 2), (-1, -2), (-1, -2), (-2, -2), (-2, 2)]) # Grow to 5x3
        danger_offsets.extend([(-3, -2), (-3, -1), (-3, 0), (-3, 1), (-3, 2)]) # Grow to 5x4
        danger_offsets.extend([(-4, -2), (-4, -1), (-4, 0), (-4, 1), (-4, 2)]) # Grow to 5x5
        for rel_offset in danger_offsets:
            abs_offset = rel_to_abs_offset(rel_offset)
            new_coords = add_coords(head, abs_offset)
            danger = check_out_bounds(new_coords, self.config) or check_collision(self.grid, new_coords)
            state.append(danger)

        closest_tomato = get_closest_tomato(head, self.tomatoes)
        closest_tomato_offset = (closest_tomato[0] - head[0], closest_tomato[1] - head[1])
        tomato_rel_offset = abs_to_rel_offset(closest_tomato_offset)

        food_left = tomato_rel_offset[1] < 0
        food_up = tomato_rel_offset[0] < 0
        food_right = tomato_rel_offset[1] > 0
        food_down = tomato_rel_offset[0] > 0

        state.extend([food_left, food_up, food_right, food_down])

        return torch.tensor(state, dtype=torch.float)
    
    def get_grid(self):
        return self.grid

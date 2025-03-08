import pygame
from collections import deque
from movement import *
import numpy as np
import torch
import pickle
from utils import *


class SnakeEnvironment:
    def __init__(self, config, save=False): # config must be a dictionary containing 'GRID_WIDTH' and 'GRID_HEIGHT' entries
        path1 = deque()
        path2 = deque()
        self.tomatoes = []
        self.grid = [[Square.EMPTY] * config['GRID_WIDTH'] for _ in range(config['GRID_HEIGHT'])]
        head1_dir = Direction.RIGHT
        head2_dir = Direction.LEFT
        snake1_length = 0
        snake2_length = 0

        self.snake1 = SnakeData(path1, head1_dir, snake1_length, True)
        self.snake2 = SnakeData(path2, head2_dir, snake2_length, True)

        self.config = config

        self.actions = {0:Direction.LEFT, 1:Direction.UP, 2:Direction.RIGHT, 3:Direction.DOWN}
        self.move_counter = 0
        self.truncate_limit = 5000

        num_cells = config['GRID_WIDTH'] * config['GRID_HEIGHT']
        self.snake_arr_size = num_cells // 2 + 1

        self.save = save
        if save:
            self.snake1_actions = []
            self.snake2_actions = []
            self.tomato_positions = []

        # Hard coded - starts with snake lengths of 2
        self.reset(2, 2)

    def reset(self, snake1_length=2, snake2_length=2):
        self.move_counter = 0

        self.snake1.path, self.snake2.path = deque(), deque()
        grid = [[Square.EMPTY] * self.config['GRID_WIDTH'] for _ in range(self.config['GRID_HEIGHT'])] # coordinates are (y,x)
        
        for i in range(0, snake1_length):
            self.snake1.path.appendleft((1,i))
            grid[1][i] = Square.PLAYER1
        
        for j in range(19, 19 - snake2_length, -1):
            self.snake2.path.appendleft((17, j))
            grid[17][j] = Square.PLAYER2

        self.snake1.head_dir = Direction.RIGHT
        self.snake2.head_dir = Direction.LEFT

        self.snake1.length = snake1_length
        self.snake2.length = snake2_length

        self.snake1.alive = True
        self.snake2.alive = True

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

        self.grid = grid
        return self.get_network_state()
    
    def step(self, input1, input2, update_state=True):
        old_snake1_length = self.snake1.length
        old_head = self.snake1.path[0]
        old_tomatoes = self.tomatoes

        grid, snake1, snake2, tomatoes, _, tomato_pos1, tomato_pos2 = handle_movement(game_state=self.get_game_state(), input1=input1, input2=input2, config=self.config)

        if update_state:
            self.grid, self.snake1, self.snake2, self.tomatoes = grid, snake1, snake2, tomatoes

        terminated = False
        if (not snake1.alive) and (not snake2.alive):
            terminated = True
        # if (not snake1.alive and snake1.length < snake2.length) or (not snake2.alive and snake1.length > snake2.length): #FIXME: UNCOMMENT
        #     terminated = True
        
        self.move_counter += 1
        truncated = self.move_counter >= self.truncate_limit

        # FIXME: reward handled only for snake 1 right now 
        reward1, reward2 = 0, 0
        # if terminated:
        #     reward1 += 100 if snake1.length > snake2.length else -100
        
        # if snake2 and not self.snake2:  # Reward for killing snake 2
        #     reward += 50

        # if snake1.length > old_snake1_length:  # New length is greater than old length, meaning snake ate a tomato
        #     reward1 += 10

        # if check_perpendicular_directions(input1, old_snake1_head_dir):
        #     reward1 += 1000
        reward1 += 5
        if self.snake1.length > old_snake1_length:
            reward1 += 15
        
        _, new_dist = get_closest_tomato(self.snake1.path[0], self.tomatoes)
        _, old_dist = get_closest_tomato(old_head, old_tomatoes)
        if new_dist < old_dist:
            reward1 += 1

        if self.save:
            self.snake1_actions.append(input1)
            self.snake2_actions.append(input2)
            if tomato_pos1 is not None:
                self.tomato_positions.append(tomato_pos1)
            if tomato_pos2 is not None:
                self.tomato_positions.append(tomato_pos2)
            

        return (self.get_network_state(), reward1, reward2, terminated, truncated)
        
    
    def get_game_state(self):
        return self.grid, self.snake1, self.snake2, self.tomatoes
    
    def get_network_state(self, is_snake1=True):  # FIXME: try the dumb blind snake way where it just sees danger around it
        # snake1 = np.array(self.snake1.path)
        # snake1_padding = self.snake_arr_size - snake1.shape[0]
        # if snake1_padding > 0:
        #     snake1 = np.pad(snake1, ((0, snake1_padding), (0, 0)), mode='constant', constant_values=0)

        # snake2 = np.array(self.snake2.path)
        # snake2_padding = self.snake_arr_size - snake2.shape[0]
        # if snake2_padding > 0:
        #     snake2 = np.pad(snake2, ((0, snake2_padding), (0, 0)), mode='constant', constant_values=0)

        # tomatoes = np.array(self.tomatoes)
        # if is_snake1:
        #     out = torch.tensor(np.concatenate((snake1.flatten(), snake2.flatten(), tomatoes.flatten())), dtype=torch.float)
        # else:
        #     out = torch.tensor(np.concatenate((snake2.flatten(), snake1.flatten(), tomatoes.flatten())), dtype=torch.float)
        
        # return out

        # grid_nums = [[cell.value for cell in row] for row in self.grid]
        # grid_array = np.asarray(grid_nums, dtype=np.float32)
        # concat_array = np.concatenate((grid_array.flatten(), np.array(self.snake1.path[0]).flatten(), np.array(self.snake2.path[0]).flatten()))

        head = self.snake1.path[0] if is_snake1 else self.snake2.path[0]
        left, right = add_coords(head, Direction.LEFT.value), add_coords(head, Direction.RIGHT.value)
        up, down = add_coords(head, Direction.UP.value), add_coords(head, Direction.DOWN.value)

        danger_left = check_out_bounds(left, self.config) or check_collision(self.grid, left)
        danger_up = check_out_bounds(up, self.config) or check_collision(self.grid, up)
        danger_right = check_out_bounds(right, self.config) or check_collision(self.grid, right)
        danger_down = check_out_bounds(down, self.config) or check_collision(self.grid, down)

        (tomato_dist_y, tomato_dist_x), _ = get_closest_tomato(head, self.tomatoes)

        # print(f'left: {danger_left}, up: {danger_up}, right: {danger_right}, down: {danger_down}')

        return torch.tensor([danger_left, danger_up, danger_right, danger_down, tomato_dist_x, tomato_dist_y], dtype=torch.float)
            


        # return torch.tensor(concat_array, dtype=torch.float)
    
    def save_game(self, save_path):
        print(f'Saving game to {save_path}')
        with open(save_path, "wb") as f:
            pickle.dump((self.snake1_actions, self.snake2_actions, self.tomato_positions), f)

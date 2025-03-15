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
        self.actions = {0:Direction.LEFT, 1:Direction.UP, 2:Direction.RIGHT, 3:Direction.DOWN}
        
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
            action1 = self.actions[action1]
        if type(action2) == int:
            action2 = self.actions[action2]

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
        return (self.get_portion_grid(is_snake1=True), reward1, reward2, terminated, truncated)
    
    def get_network_state(self, is_snake1=True):
        '''Returns current state of game in form (danger_left, danger_up, danger_right, danger_down, food_left, food_up, food_right, food_down)'''
        head = self.snake1.path[0] if is_snake1 else self.snake2.path[0]
        left, right = add_coords(head, Direction.LEFT.value), add_coords(head, Direction.RIGHT.value)
        up, down = add_coords(head, Direction.UP.value), add_coords(head, Direction.DOWN.value)

        danger_left = check_out_bounds(left, self.config) or check_collision(self.grid, left)
        danger_up = check_out_bounds(up, self.config) or check_collision(self.grid, up)
        danger_right = check_out_bounds(right, self.config) or check_collision(self.grid, right)
        danger_down = check_out_bounds(down, self.config) or check_collision(self.grid, down)

        closest_tomato = get_closest_tomato(head, self.tomatoes)

        food_left = closest_tomato[1] < head[1]
        food_up = closest_tomato[0] < head[0]
        food_right = closest_tomato[1] > head[1]
        food_down = closest_tomato[0] > head[0]

        return torch.tensor([danger_left, danger_up, danger_right, danger_down, food_left, food_up, food_right, food_down], dtype=torch.float)
    
    def get_grid(self):
        return self.grid

    def get_portion_grid(self, is_snake1):
        '''
        Return a 5x5 grid centered on head_pos as a torch tensor with 1 channel.
        If grid portion goes out-of-bounds, fill with danger (-1).
        The returned tensor has shape (1, 5, 5) so that it can be directly input into the CNN.
        '''
        head_pos = self.snake1.path[0] if is_snake1 else self.snake2.path[0]
        head_y, head_x = head_pos
        top_left_y, top_left_x = head_y - 2, head_x - 2

        # 5x5 grid where each cell is a list of three values (R, G, B)
        portion_grid = []
        for y in range(top_left_y, top_left_y + 5):
            row = []
            for x in range(top_left_x, top_left_x + 5):
                if check_out_bounds((y, x), self.config) or check_collision(self.grid, (y, x)):
                    val = -1
                else:
                    val = self.grid[y][x].value

                row.append([val, val, val])
            portion_grid.append(row)

        grid_tensor = torch.tensor(portion_grid, dtype=torch.float32)

        # grid_tensor = grid_tensor.unsqueeze(0) # (1, 5, 5)
        grid_tensor = grid_tensor.permute(2, 0, 1)
        grid_tensor = grid_tensor.unsqueeze(0) # (1, 1, 5, 5)

        blind_tensor = self.get_network_state(is_snake1=is_snake1)  

        blind_tensor = blind_tensor.view(1, 8, 1, 1) # shape becomes (1, 8, 1, 1)
        blind_tensor = blind_tensor.expand(1, 8, grid_tensor.size(2), grid_tensor.size(3))  # (1, 8, 5, 5)

        combined_tensor = torch.cat((grid_tensor, blind_tensor), dim=1)  # shape: (1, 11, 5, 5)

        return combined_tensor

    def check_dies(self, direction, is_snake1):
        if check_perpendicular_directions(self.snake1.head_dir, direction):
            self.snake1.head_dir = direction
        if check_perpendicular_directions(self.snake2.head_dir, direction):
            self.snake2.head_dir = direction
        # Checks if snakes went out of bounds or collided with something
        _, self.snake1.alive = check_ob_collisions(self.grid, self.snake1.path, self.snake1.head_dir, True, self.config)
        _, self.snake2.alive = check_ob_collisions(self.grid, self.snake2.path, self.snake2.head_dir, True, self.config)

        return self.snake1.alive if is_snake1 else self.snake2.alive
import pygame
from collections import deque
from movement import Direction, Square, handle_movement


class SnakeEnvironment:
    def __init__(self, config): # config must be a dictionary containing 'GRID_WIDTH' and 'GRID_HEIGHT' entries
        self.snake1 = deque()
        self.snake2 = deque()
        self.tomatoes = []
        self.grid = [[Square.EMPTY] * config['GRID_WIDTH'] for _ in range(config['GRID_HEIGHT'])]
        self.head1_dir = None
        self.head2_dir = None
        self.snake1_length = 0
        self.snake2_length = 0

        self.config = config
        # Hard coded - starts with snake lengths of 2
        self.reset(2, 2, config)

        self.actions = {0:'LEFT', 1:'STRAIGHT', 2:'RIGHT'}
        self.move_counter = 0
        self.truncate_limit = 50000

    def reset(self, snake1_length, snake2_length, config):
        snake1, snake2 = deque(), deque()
        grid = [[Square.EMPTY] * config['GRID_WIDTH'] for _ in range(config['GRID_HEIGHT'])] # coordinates are (y,x)
        
        for i in range(1, snake1_length+1):
            snake1.appendleft((4,i))
            grid[4][i] = Square.PLAYER1
        
        for j in range(15, 15 - snake2_length, -1):
            snake2.appendleft((5, j))
            grid[5][j] = Square.PLAYER2

        head1_dir = Direction.RIGHT
        head2_dir = Direction.LEFT

        tomatoes = []
        tomatoes.append((6,4))
        tomatoes.append((0,1))
        tomatoes.append((8,6))
        tomatoes.append((17,2))
        tomatoes.append((16,18))
        for coord in tomatoes:
            row = coord[0]
            col = coord[1]
            grid[row][col] = Square.TOMATO

        self.grid, self.snake1, self.head1_dir, self.snake1_length, self.snake2, self.head2, self.snake2_length = grid, snake1, head1_dir, snake1_length, snake2, head2_dir, snake2_length
        return self.get_state()
    
    def step(self, game_state, input1, input2, config):
        self.grid, self.snake1, self.head1_dir, self.snake1_length, self.snake2, self.head2_dir, self.snake2_length = handle_movement(game_state=game_state, input1=input1, input2=input2, config=config)

        new_state = self.get_state()

        # FIXME: how to handle reward? 
        reward = 0

        terminated = True
        if self.snake1 is None and self.snake2 is None:
            terminated = False
        if (self.snake1 is None and self.snake1_length < self.snake2_length) or (self.snake2 is None and self.snake1_length > self.snake2_length):
            terminated = False
        
        truncated = False
        self.counter += 1
        if self.counter >= self.move_counter:
            truncated = True

        return (new_state, reward, terminated, truncated)
        
        
        
    
    def get_state(self):
        return (self.grid, self.snake1, self.head1_dir, self.snake1_length, self.snake2, self.head2_dir, self.snake2_length)

    
import pygame
import random
from enum import Enum

class Direction(Enum):
    LEFT = (0, -1)
    UP = (-1, 0)
    RIGHT = (0, 1)
    DOWN = (1, 0)

class Square(Enum):
    EMPTY = 0
    PLAYER1 = 1
    PLAYER2 = 2
    TOMATO = 3

def get_square(grid, pos):
    return grid[pos[0]][pos[1]]

def set_square(grid, pos, type):
    grid[pos[0]][pos[1]] = type

def add_tomato(grid, config):
    while True:
        row = random.choice(range(config['GRID_HEIGHT']))
        col = random.choice(range(config['GRID_WIDTH']))
        if grid[row][col] == Square.EMPTY:
            grid[row][col] = Square.TOMATO
            break

def check_ob_collisions(grid, snake, head_dir, config):
        if snake is None:
            return grid, None
        
        # Calculate new head position
        head_pos = snake[0]
        new_head_pos = tuple(a + b for a, b in zip(head_pos, head_dir.value))  # Element-wise addition

        # Check for out of bounds
        out_bounds = new_head_pos[0] < 0 or new_head_pos[0] >= config['GRID_HEIGHT'] or new_head_pos[1] < 0 or new_head_pos[1] >= config['GRID_WIDTH']
        if out_bounds:
            return grid, None
        
        # Check for collision
        new_square = get_square(grid, new_head_pos)
        snake_crashed = new_square in [Square.PLAYER1, Square.PLAYER2]
        if snake_crashed:
            return grid, None
        
        return (grid, snake)

def update_snake(grid, snake, head_dir, square):
        if snake is None:
            return grid, None, False

        head_pos = snake[0]
        new_head_pos = tuple(a + b for a, b in zip(head_pos, head_dir.value))  # Element-wise addition

        # Check for eating tomatos
        new_square = get_square(grid, new_head_pos)
        snake_ate = new_square == Square.TOMATO
        if not snake_ate:  # Shrink tail unless snake ate a tomato
            tail_pos = snake.pop()
            set_square(grid, tail_pos, Square.EMPTY)
        
        # Grow in correct direction
        set_square(grid, new_head_pos, square)
        snake.appendleft(new_head_pos)

        return grid, snake, snake_ate

def handle_input(key, head1_dir, head2_dir):
    # Get direction input from keyboard, default to going straight
    if key[pygame.K_a]:
        input1 = Direction.LEFT
    elif key[pygame.K_s]:
        input1 = Direction.DOWN
    elif key[pygame.K_d]:
        input1 = Direction.RIGHT
    elif key[pygame.K_w]:
        input1 = Direction.UP
    else:
        input1 = head1_dir
    
    if key[pygame.K_LEFT]:
        input2 = Direction.LEFT
    elif key[pygame.K_DOWN]:
        input2 = Direction.DOWN
    elif key[pygame.K_RIGHT]:
        input2 = Direction.RIGHT
    elif key[pygame.K_UP]:
        input2 = Direction.UP
    else:
        input2 = head2_dir
    
    return input1, input2

def handle_movement(game_state, key, config):
    grid, snake1, head1_dir, snake2, head2_dir = game_state

    input1, input2 = handle_input(key, head1_dir, head2_dir)
    
    def check_perpendicular_directions(dir1, dir2):
        return (dir1 in [Direction.LEFT, Direction.RIGHT]) != (dir2 in [Direction.LEFT, Direction.RIGHT])
    
    # Handle snake turning
    if check_perpendicular_directions(head1_dir, input1):
        head1_dir = input1
    if check_perpendicular_directions(head2_dir, input2):
        head2_dir = input2

    grid, snake1 = check_ob_collisions(grid, snake1, head1_dir, config)
    grid, snake2 = check_ob_collisions(grid, snake2, head2_dir, config)
        
    grid, snake1, snake1_ate = update_snake(grid, snake1, head1_dir, Square.PLAYER1)
    grid, snake2, snake2_ate = update_snake(grid, snake2, head2_dir, Square.PLAYER2)

    # Add new tomatos if any were eaten
    if snake1_ate:
        add_tomato(grid, config)
    if snake2_ate:
        add_tomato(grid, config)

    # Return updated game state
    game_state = (grid, snake1, head1_dir, snake2, head2_dir)
    return game_state
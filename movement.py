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

def handle_movement(game_state, key, config):
    grid, snake1, head1_dir, snake2, head2_dir, game_end = game_state

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
    
    if key[pygame.K_UP]:
        input2 = Direction.UP
    elif key[pygame.K_DOWN]:
        input2 = Direction.DOWN
    elif key[pygame.K_RIGHT]:
        input2 = Direction.RIGHT
    elif key[pygame.K_LEFT]:
        input2 = Direction.LEFT
    else:
        input2 = head2_dir

    def check_perpendicular_directions(dir1, dir2):
        return (dir1 in [Direction.LEFT, Direction.RIGHT]) != (dir2 in [Direction.LEFT, Direction.RIGHT])
    
    # Handle snake turning
    if check_perpendicular_directions(head1_dir, input1):
        head1_dir = input1
    if check_perpendicular_directions(head2_dir, input2):
        head2_dir = input2

    # Calculate new head position for each snake
    head1_pos = snake1[0]
    head2_pos = snake2[0]
    new_head1_pos = tuple(a + b for a, b in zip(head1_pos, head1_dir.value))  # Element-wise addition
    new_head2_pos = tuple(a + b for a, b in zip(head2_pos, head2_dir.value))

    # Check for out of bounds
    snake1_out_bounds = new_head1_pos[0] < 0 or new_head1_pos[0] >= config['GRID_HEIGHT'] or new_head1_pos[1] < 0 or new_head1_pos[1] >= config['GRID_WIDTH']
    snake2_out_bounds = new_head2_pos[0] < 0 or new_head2_pos[0] >= config['GRID_HEIGHT'] or new_head2_pos[1] < 0 or new_head2_pos[1] >= config['GRID_WIDTH']
    if snake1_out_bounds or snake2_out_bounds:
        print('Snake out of bounds')
        return (grid, snake1, head1_dir, snake2, head2_dir, True)

    # Check for eating tomatos
    new_square1 = get_square(grid, new_head1_pos)
    new_square2 = get_square(grid, new_head2_pos)
    snake1_ate = new_square1 == Square.TOMATO
    snake2_ate = new_square2 == Square.TOMATO

    # Check for collision
    snake1_crashed = new_square1 in [Square.PLAYER1, Square.PLAYER2]
    snake2_crashed = new_square2 in [Square.PLAYER1, Square.PLAYER2]
    if snake1_crashed or snake2_crashed:
        print('Snake crashed')
        return (grid, snake1, head1_dir, snake2, head2_dir, True)

    # Shrink tail unless snake ate a tomato
    if not snake1_ate:
        tail1_pos = snake1.pop()
        set_square(grid, tail1_pos, Square.EMPTY)
    if not snake2_ate:
        tail2_pos = snake2.pop()
        set_square(grid, tail2_pos, Square.EMPTY)

    # Grow snakes in correct direction
    set_square(grid, new_head1_pos, Square.PLAYER1)
    set_square(grid, new_head2_pos, Square.PLAYER2)
    snake1.appendleft(new_head1_pos)
    snake2.appendleft(new_head2_pos)

    # Add new tomatos if any were eaten
    if snake1_ate:
        add_tomato(grid, config)
    if snake2_ate:
        add_tomato(grid, config)
    
    # Return updated game state
    game_state = (grid, snake1, head1_dir, snake2, head2_dir, False)
    return game_state
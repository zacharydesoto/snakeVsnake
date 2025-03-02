import pygame
import random
from enum import Enum
from collections import defaultdict

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
        return grid, None, None
    
    # Calculate new head position
    head_pos = snake[0]
    new_head_pos = tuple(a + b for a, b in zip(head_pos, head_dir.value))  # Element-wise addition

    # Check for out of bounds
    out_bounds = new_head_pos[0] < 0 or new_head_pos[0] >= config['GRID_HEIGHT'] or new_head_pos[1] < 0 or new_head_pos[1] >= config['GRID_WIDTH']
    if out_bounds:
        return grid, None, None
    
    # Check for collision
    new_square = get_square(grid, new_head_pos)
    snake_crashed = new_square in [Square.PLAYER1, Square.PLAYER2]
    if snake_crashed:
        return grid, None, None
    
    return (grid, snake, new_head_pos)

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

def one_player_input(left, down, right, up, prevleft, prevdown, prevright, prevup, old_input):
    if not (left or down or right or up):
        return old_input
    
    if left and not (down or right or up):
        return Direction.LEFT
    if down and not (left or right or up):
        return Direction.DOWN
    if right and not (down or left or up):
        return Direction.RIGHT
    if up and not (down or right or left):
        return Direction.UP
    
    if left and not prevleft:
        return Direction.LEFT
    if down and not prevdown:
        return Direction.DOWN
    if right and not prevright:
        return Direction.RIGHT
    if up and not prevup:
        return Direction.UP

    # 2 or more pressed and same as last time
    # should return old input
    return old_input
    


def handle_input(key, prev, input1, input2):
    prevleft1 = prev['left1']
    prevdown1 = prev['down1']
    prevright1 = prev['right1']
    prevup1 = prev['up1']
    prevleft2 = prev['left2']
    prevdown2 = prev['down2']
    prevright2 = prev['right2']
    prevup2 = prev['up2']

    prev = defaultdict(bool)

    left1, down1, right1, up1 = False, False, False, False
    left2, down2, right2, up2 = False, False, False, False
    # Get direction input from keyboard, default to going straight
    if key[pygame.K_a]:
        left1 = True
        prev['left1'] = True
    if key[pygame.K_s]:
        down1 = True
        prev['down1'] = True
    if key[pygame.K_d]:
        right1 = True
        prev['right1'] = True
    if key[pygame.K_w]:
        up1 = True
        prev['up1'] = True
    
    if key[pygame.K_LEFT]:
        left2 = True
        prev['left2'] = True
    if key[pygame.K_DOWN]:
        down2 = True
        prev['down2'] = True
    if key[pygame.K_RIGHT]:
        right2 = True
        prev['right2'] = True
    if key[pygame.K_UP]:
        up2 = True
        prev['up2'] = True

    input1 = one_player_input(left1, down1, right1, up1, prevleft1, prevdown1, prevright1, prevup1, input1)
    input2 = one_player_input(left2, down2, right2, up2, prevleft2, prevdown2, prevright2, prevup2, input2)
    
    return input1, input2, prev

def handle_movement(game_state, input1, input2, config):
    grid, snake1, head1_dir, snake1_length, snake2, head2_dir, snake2_length = game_state
    
    def check_perpendicular_directions(dir1, dir2):
        return (dir1 in [Direction.LEFT, Direction.RIGHT]) != (dir2 in [Direction.LEFT, Direction.RIGHT])
    
    # Handle snake turning
    if check_perpendicular_directions(head1_dir, input1):
        head1_dir = input1
    if check_perpendicular_directions(head2_dir, input2):
        head2_dir = input2

    grid, snake1, head1 = check_ob_collisions(grid, snake1, head1_dir, config)
    grid, snake2, head2 = check_ob_collisions(grid, snake2, head2_dir, config)

    if head1 is not None and head2 is not None and head1 == head2:
        snake1 = None
        snake2 = None
        grid[head1[0]][head1[1]] = Square.PLAYER1
        
    grid, snake1, snake1_ate = update_snake(grid, snake1, head1_dir, Square.PLAYER1)
    grid, snake2, snake2_ate = update_snake(grid, snake2, head2_dir, Square.PLAYER2)

    # Add new tomatos if any were eaten
    if snake1_ate:
        add_tomato(grid, config)
        snake1_length += 1
    if snake2_ate:
        add_tomato(grid, config)
        snake2_length += 1

    # Return updated game state
    game_state = (grid, snake1, head1_dir, snake1_length, snake2, head2_dir, snake2_length)
    return game_state
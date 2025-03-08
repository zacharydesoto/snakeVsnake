import pygame
import random
from enum import Enum
from collections import defaultdict
from utils import *

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

class SnakeData():
    def __init__(self, path, head_dir, length, alive):
        self.path = path
        self.head_dir = head_dir
        self.length = length
        self.alive = alive

def add_tomato(grid, config, pos=None):
    if pos is None:
        while True:
            row = random.choice(range(config['GRID_HEIGHT']))
            col = random.choice(range(config['GRID_WIDTH']))
            if grid[row][col] == Square.EMPTY:
                grid[row][col] = Square.TOMATO
                break
        return (row, col)
    else:
        grid[pos[0]][pos[1]] = Square.TOMATO
        return pos

def check_ob_collisions(grid, path, head_dir, alive, config):
    if not alive:
        return grid, None, False
    
    # Calculate new head position
    head_pos = path[0]
    new_head_pos = tuple(a + b for a, b in zip(head_pos, head_dir.value))  # Element-wise addition

    # Check for out of bounds
    if check_out_bounds(new_head_pos, config):
        return grid, None, False
    
    # Check for collision
    if check_collision(grid, new_head_pos):
        return grid, None, False
    
    return grid, new_head_pos, True

def update_snake(grid, snake, square):
    if not snake.alive:
        return grid, snake.path, False

    head_pos = snake.path[0]
    new_head_pos = add_coords(head_pos, snake.head_dir.value)

    # Check for eating tomatos
    new_square = get_square(grid, new_head_pos)
    snake_ate = new_square == Square.TOMATO
    if not snake_ate:  # Shrink tail unless snake ate a tomato
        tail_pos = snake.path.pop()
        set_square(grid, tail_pos, Square.EMPTY)
    
    # Grow in correct direction
    set_square(grid, new_head_pos, square)
    snake.path.appendleft(new_head_pos)

    return grid, snake.path, snake_ate

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

def handle_movement(game_state, input1, input2, config, tomato_positions=None):
    grid, snake1, snake2, tomatoes = game_state
    head1_dir, head2_dir = snake1.head_dir, snake2.head_dir
    
    # Handle snake turning
    if check_perpendicular_directions(head1_dir, input1):
        head1_dir = input1
    if check_perpendicular_directions(head2_dir, input2):
        head2_dir = input2
    
    snake1.head_dir = head1_dir
    snake2.head_dir = head2_dir

    grid, head1, snake1.alive = check_ob_collisions(grid, snake1.path, head1_dir, snake1.alive, config)
    grid, head2, snake2.alive = check_ob_collisions(grid, snake2.path, head2_dir, snake2.alive, config)

    if head1 is not None and head2 is not None and head1 == head2:
        snake1.alive = False
        snake2.alive = False
        grid[head1[0]][head1[1]] = Square.PLAYER1
        
    grid, snake1.path, snake1_ate = update_snake(grid, snake1, Square.PLAYER1)
    grid, snake2.path, snake2_ate = update_snake(grid, snake2, Square.PLAYER2)

    # Add new tomatos if any were eaten
    tomato_pos1,tomato_pos2 = None, None
    if snake1_ate:
        tomatoes.remove(head1)
        if tomato_positions is not None:
            tomato_pos1 = tomato_positions.pop[0]
            add_tomato(grid, config, pos=tomato_pos1)
        else:
            tomato_pos1 = add_tomato(grid, config)
        tomatoes.append(tomato_pos1)
        snake1.length += 1
    if snake2_ate:
        tomatoes.remove(head2)
        if tomato_positions is not None:
            tomato_pos2 = tomato_positions.pop[0]
            add_tomato(grid, config, pos=tomato_pos2)
        else:
            tomato_pos2 = add_tomato(grid, config)
        tomatoes.append(tomato_pos2)
        snake2.length += 1

    # Return updated game state
    game_state = (grid, snake1, snake2, tomatoes, tomato_positions, tomato_pos1, tomato_pos2)
    return game_state

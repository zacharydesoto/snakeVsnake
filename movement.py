import pygame
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

def handle_movement(game_state, key, config):
    grid, snake1, head1_dir, snake2, head2_dir, game_end = game_state

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
    
    if check_perpendicular_directions(head1_dir, input1):
        head1_dir = input1
    if check_perpendicular_directions(head2_dir, input2):
        head2_dir = input2

    head1_pos = snake1[0]
    head2_pos = snake2[0]
    new_head1_pos = head1_pos + head1_dir.value
    new_head2_pos = head2_pos + head2_dir.value
    tail1_pos = snake1.popleft()
    tail2_pos = snake2.popleft()

    if new_head1_pos[0] < 0 or new_head1_pos[0] > config['GRID_HEIGHT'] or new_head1_pos[1] < 0 or new_head1_pos[1] > config['GRID_WIDTH']:
        game_end = True
    if new_head2_pos[0] < 0 or new_head2_pos[0] > config['GRID_HEIGHT'] or new_head2_pos[1] < 0 or new_head2_pos[1] > config['GRID_WIDTH']:
        game_end = True

    print(new_head1_pos)
    print(new_head2_pos)
    grid[new_head1_pos[0]][new_head1_pos[1]] = Square.PLAYER1
    grid[new_head2_pos[0]][new_head2_pos[1]] = Square.PLAYER2
    snake1.append(new_head1_pos)
    snake2.append(new_head2_pos)

    # print(head1_pos)
    # print(head2_pos)
    
    game_state = (grid, snake1, head1_dir, snake2, head2_dir, game_end)
    return game_state
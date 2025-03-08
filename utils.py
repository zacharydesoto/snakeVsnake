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

def grid_to_screen(grid_x, grid_y, config):
    screen_x = grid_x * config['SCREEN_WIDTH'] / config['GRID_WIDTH'] + config['MARGIN']
    screen_y = grid_y * config['SCREEN_HEIGHT'] / config['GRID_HEIGHT'] + config['MARGIN'] + config['TOP_MARGIN']
    return screen_x, screen_y

def check_perpendicular_directions(dir1, dir2):
    return (dir1 in [Direction.LEFT, Direction.RIGHT]) != (dir2 in [Direction.LEFT, Direction.RIGHT])

def check_out_bounds(pos, config):
    return pos[0] < 0 or pos[0] >= config['GRID_HEIGHT'] or pos[1] < 0 or pos[1] >= config['GRID_WIDTH']

def check_collision(grid, pos):
    return grid[pos[0]][pos[1]] in [Square.PLAYER1, Square.PLAYER2]

def add_coords(coord1, coord2):
    return (coord1[0] + coord2[0], coord1[1] + coord2[1])

def get_square(grid, pos):
    return grid[pos[0]][pos[1]]

def set_square(grid, pos, type):
    grid[pos[0]][pos[1]] = type

def calculate_distance(coord1, coord2):
    return (coord2[0] - coord1[0]) + (coord2[1] - coord1[1])

def get_closest_tomato(head, tomatoes):
    closest_dist, closest_coords = float('inf'), (0, 0)
    for coords in tomatoes:
        if calculate_distance(coords, head) < closest_dist:
            closest_dist = calculate_distance(coords, head)
            closest_coords = coords
    
    tomato_dist_y = closest_coords[0] - head[0]
    tomato_dist_x = closest_coords[1] - head[1]
    return (tomato_dist_y, tomato_dist_x), closest_dist

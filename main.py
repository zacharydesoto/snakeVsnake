import pygame
from enum import Enum
from collections import deque

from utils import *
from movement import *

pygame.init()

config = {}
SCREEN_WIDTH, SCREEN_HEIGHT  = 800, 800  # Make sure screen width and height are multiples of grid width and height
GRID_WIDTH, GRID_HEIGHT = 10, 10
MARGIN = 10
SQUARE_WIDTH, SQUARE_HEIGHT = SCREEN_WIDTH/GRID_WIDTH, SCREEN_HEIGHT/GRID_HEIGHT
config['SCREEN_WIDTH'], config['SCREEN_HEIGHT'] = SCREEN_WIDTH, SCREEN_HEIGHT
config['GRID_WIDTH'], config['GRID_HEIGHT'] = GRID_WIDTH, GRID_HEIGHT
config['MARGIN'] = MARGIN

screen = pygame.display.set_mode((SCREEN_WIDTH+2*MARGIN, SCREEN_HEIGHT+2*MARGIN))
clock = pygame.time.Clock()

pygame.display.set_caption('SnakeRL')

grid = [[Square.EMPTY] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
grid[4][2] = Square.PLAYER1
grid[4][1] = Square.PLAYER1
grid[4][6] = Square.PLAYER2
grid[4][7] = Square.PLAYER2
grid[4][4] = Square.TOMATO
grid[0][1] = Square.TOMATO

snake1, snake2 = deque(), deque()
snake1.append((4, 2))
snake1.append((4, 1))
snake2.append((4, 6))
snake2.append((4, 7))
head1_dir = Direction.RIGHT
head2_dir = Direction.LEFT

game_state = (grid, snake1, head1_dir, snake2, head2_dir, False)

run = True
while run:
    key = pygame.key.get_pressed()

    game_state = handle_movement(game_state, key, config)
    grid, snake1, head1_dir, snake2, head2_dir, game_end = game_state
    if game_end:
        run = False

    screen.fill((0, 0, 0))

    # Outputs grid to screen
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            x, y = grid_to_screen(col, row, config)
            if grid[row][col] == Square.PLAYER1:
                pygame.draw.rect(screen, (0, 255, 0), (x, y, SQUARE_WIDTH, SQUARE_HEIGHT))
            elif grid[row][col] == Square.PLAYER2:
                pygame.draw.rect(screen, (0, 0, 255), (x, y, SQUARE_WIDTH, SQUARE_HEIGHT))
            elif grid[row][col] == Square.TOMATO:
                pygame.draw.rect(screen, (255, 0, 0), (x, y, SQUARE_WIDTH, SQUARE_HEIGHT))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    
    pygame.display.update()

    clock.tick(1)

pygame.quit()
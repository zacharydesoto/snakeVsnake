import pygame
from enum import Enum

from utils import *
from movement import *

pygame.init()

config = {}
SCREEN_WIDTH, SCREEN_HEIGHT  = 800, 800
GRID_WIDTH, GRID_HEIGHT = 10, 10
config['SCREEN_WIDTH'], config['SCREEN_HEIGHT'] = SCREEN_WIDTH, SCREEN_HEIGHT
config['GRID_WIDTH'], config['GRID_HEIGHT'] = GRID_WIDTH, GRID_HEIGHT

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

pygame.display.set_caption('SnakeRL')


SQUARE_WIDTH, SQUARE_HEIGHT = SCREEN_WIDTH/GRID_WIDTH, SCREEN_HEIGHT/GRID_HEIGHT

class Square(Enum):
    EMPTY = 0
    PLAYER1 = 1
    PLAYER2 = 2
    TOMATO = 3

grid = [[Square.EMPTY] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
grid[4][2] = Square.PLAYER1
grid[4][1] = Square.PLAYER1
grid[4][6] = Square.PLAYER2
grid[4][7] = Square.PLAYER2
grid[4][4] = Square.TOMATO

run = True
while run:
    key = pygame.key.get_pressed()
    grid = handle_movement(grid, key)

    screen.fill((0, 0, 0))

    RED = (255, 0, 0)
    BLUE1 = (0, 0, 255)
    BLUE2 = (0, 100, 255)
    GREEN1 = (1, 135, 8)
    GREEN2 = (0, 255, 0)
    MARGIN = 10

    # Outputs grid to screen
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            x, y = grid_to_screen(col, row, config)
            if grid[row][col] == Square.PLAYER1:
                pygame.draw.rect(screen, GREEN1, (x, y, SQUARE_WIDTH, SQUARE_HEIGHT))
                pygame.draw.rect(screen, GREEN2, (x+MARGIN, y+MARGIN, SQUARE_WIDTH-2*MARGIN, SQUARE_HEIGHT-2*MARGIN))
            elif grid[row][col] == Square.PLAYER2:
                pygame.draw.rect(screen, BLUE1, (x, y, SQUARE_WIDTH, SQUARE_HEIGHT))
                pygame.draw.rect(screen, BLUE2, (x+MARGIN, y+MARGIN, SQUARE_WIDTH-2*MARGIN, SQUARE_HEIGHT-2*MARGIN))
            elif grid[row][col] == Square.TOMATO:
                pygame.draw.rect(screen, RED, (x, y, SQUARE_WIDTH, SQUARE_HEIGHT))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    
    pygame.display.update()

    clock.tick(60)

pygame.quit()
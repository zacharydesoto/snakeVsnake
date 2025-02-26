import pygame
from enum import Enum
from collections import deque

from utils import *
from movement import *

pygame.init()

# Set up variables for screen and grid
config = {}
SCREEN_WIDTH, SCREEN_HEIGHT  = 800, 800  # Make sure screen width and height are multiples of grid width and height
GRID_WIDTH, GRID_HEIGHT = 20, 20
MARGIN = 10
SQUARE_WIDTH, SQUARE_HEIGHT = SCREEN_WIDTH/GRID_WIDTH, SCREEN_HEIGHT/GRID_HEIGHT
config['SCREEN_WIDTH'], config['SCREEN_HEIGHT'] = SCREEN_WIDTH, SCREEN_HEIGHT
config['GRID_WIDTH'], config['GRID_HEIGHT'] = GRID_WIDTH, GRID_HEIGHT
config['MARGIN'] = MARGIN
FRAMERATE = 60
TICKS_PER_S = 5

screen = pygame.display.set_mode((SCREEN_WIDTH+2*MARGIN, SCREEN_HEIGHT+2*MARGIN))
clock = pygame.time.Clock()

pygame.display.set_caption('SnakeRL')
pygame.font.init()

# Initialize grid and snakes
grid = [[Square.EMPTY] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
grid[4][2] = Square.PLAYER1
grid[4][1] = Square.PLAYER1
grid[5][15] = Square.PLAYER2
grid[5][14] = Square.PLAYER2
grid[6][4] = Square.TOMATO
grid[0][1] = Square.TOMATO
grid[8][6] = Square.TOMATO
grid[17][2] = Square.TOMATO
grid[16][18] = Square.TOMATO

snake1, snake2 = deque(), deque()
snake1.appendleft((4, 1))
snake1.appendleft((4, 2))
snake2.appendleft((5, 15))
snake2.appendleft((5, 14))
head1_dir = Direction.RIGHT
head2_dir = Direction.LEFT

game_state = (grid, snake1, head1_dir, snake2, head2_dir)

count = 1
input1, input2 = Direction.RIGHT, Direction.LEFT
prev = defaultdict(bool)

run = True
while run:
    key = pygame.key.get_pressed()

    # Handle input 60 times a second
    if count < FRAMERATE // TICKS_PER_S:
        input1, input2, prev = handle_input(key, prev, input1, input2)
        count += 1
    else:
        # Update game state based on player input
        count = 1
        game_state = handle_movement(game_state, input1, input2, config)
        grid, snake1, head1_dir, snake2, head2_dir = game_state
        if snake1 is None and snake2 is None:
            run = False

    # Clear old output of screen
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

    # Event handler, currently just for closing game
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    
    # Refreshes display
    pygame.display.update()

    # Sets frame rate
    clock.tick(FRAMERATE)

font = pygame.font.SysFont('ubuntusans', 100)
text_surface = font.render('Jonk Peralta', False, (25, 136, 191))
screen.blit(text_surface, (150, 350))
pygame.display.update()

pygame.time.wait(2000)
pygame.quit()
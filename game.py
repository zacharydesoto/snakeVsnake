import pygame
from enum import Enum
from collections import deque

from utils import *
from movement import *

def two_player_game():

    pygame.init()

    # Set up variables for screen and grid
    config = {}
    SCREEN_WIDTH, SCREEN_HEIGHT  = 800, 800  # Make sure screen width and height are multiples of grid width and height
    GRID_WIDTH, GRID_HEIGHT = 20, 20
    MARGIN = 10
    TOP_MARGIN = 100
    BORDER_WIDTH = 10
    SQUARE_WIDTH, SQUARE_HEIGHT = SCREEN_WIDTH/GRID_WIDTH, SCREEN_HEIGHT/GRID_HEIGHT
    config['SCREEN_WIDTH'], config['SCREEN_HEIGHT'] = SCREEN_WIDTH, SCREEN_HEIGHT
    config['GRID_WIDTH'], config['GRID_HEIGHT'] = GRID_WIDTH, GRID_HEIGHT
    config['MARGIN'] = MARGIN
    config['TOP_MARGIN'] = TOP_MARGIN
    FRAMERATE = 60
    TICKS_PER_S = 5

    screen = pygame.display.set_mode((SCREEN_WIDTH+2*MARGIN, SCREEN_HEIGHT+2*MARGIN + TOP_MARGIN))
    clock = pygame.time.Clock()

    pygame.display.set_caption('SnakeRL')
    pygame.font.init()

    # Initialize grid and snakes
    grid = [[Square.EMPTY] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
    grid[1][1] = Square.PLAYER1
    grid[1][0] = Square.PLAYER1
    grid[17][18] = Square.PLAYER2
    grid[17][19] = Square.PLAYER2
    grid[2][2] = Square.TOMATO
    grid[16][16] = Square.TOMATO
    grid[10][10] = Square.TOMATO
    grid[2][18] = Square.TOMATO
    grid[16][2] = Square.TOMATO

    snake1, snake2 = deque(), deque()
    snake1.append((1, 1))
    snake1.append((1, 0))
    snake2.append((17, 18))
    snake2.append((17, 19))

    head1_dir = Direction.RIGHT
    head2_dir = Direction.LEFT
    snake1_length = 2
    snake2_length = 2

    game_state = (grid, snake1, head1_dir, snake1_length, snake2, head2_dir, snake2_length)

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
            grid, snake1, head1_dir, snake1_length, snake2, head2_dir, snake2_length = game_state
            if snake1 is None and snake2 is None:
                run = False
            if (snake1 is None and snake1_length < snake2_length) or (snake2 is None and snake1_length > snake2_length):
                run = False

        # Clear old output of screen
        screen.fill((0, 0, 0))

        RED = (255, 0, 0)
        BLUE1 = (0, 0, 255)
        BLUE2 = (0, 100, 255)
        GREEN1 = (1, 135, 8)
        GREEN2 = (0, 255, 0)
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)

        pygame.draw.rect(screen, WHITE, (MARGIN - BORDER_WIDTH/2, MARGIN + TOP_MARGIN - BORDER_WIDTH/2, SCREEN_WIDTH + BORDER_WIDTH, SCREEN_HEIGHT + BORDER_WIDTH))
        pygame.draw.rect(screen, BLACK, (MARGIN, MARGIN + TOP_MARGIN, SCREEN_WIDTH, SCREEN_HEIGHT))

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

        length_font = pygame.font.SysFont('ubuntusans', 50)
        text_surface = length_font.render(f'Green Snake: {snake1_length}     Blue Snake: {snake2_length}', False, (25, 136, 191))
        screen.blit(text_surface, (40, 0))

        # Event handler, currently just for closing game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        
        # Refreshes display
        pygame.display.update()

        # Sets frame rate
        clock.tick(FRAMERATE)

    if snake1_length > snake2_length:
        end_text = "Green Snake Wins!"
    elif snake2_length > snake1_length:
        end_text = "Blue Snake Wins!"
    else:
        end_text = "It's a Tie!"

    length_font = pygame.font.SysFont('ubuntusans', 75)
    text_surface = length_font.render(end_text, False, WHITE)
    screen.blit(text_surface, (200, 450))
    pygame.display.update()

    pygame.time.wait(3000)
    pygame.quit()
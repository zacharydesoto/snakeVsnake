import pygame
from collections import deque

from utils import *
from movement import *
import pickle

def two_player_game(saved_game=None):

    pygame.init()

    if saved_game is not None:
        snake1_actions, snake2_actions, tomato_positions = saved_game
        # print(snake1_actions)
        print(f'Snake 1: {len(snake1_actions)} Snake 2: {len(snake2_actions)}')
        load_game = True

    # Set up variables for screen and grid
    config = {}
    SCREEN_WIDTH, SCREEN_HEIGHT  = 500, 500  # Make sure screen width and height are multiples of grid width and height
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

    tomatoes = [(2, 2), (16, 16), (10, 10), (2, 18), (16, 2)]

    path1, path2 = deque(), deque()
    path1.append((1, 1))
    path1.append((1, 0))
    path2.append((17, 18))
    path2.append((17, 19))

    head1_dir = Direction.RIGHT
    head2_dir = Direction.LEFT
    snake1_length = 2
    snake2_length = 2

    snake1 = SnakeData(path1, head1_dir, snake1_length, True)
    snake2 = SnakeData(path2, head2_dir, snake2_length, True)

    game_state = (grid, snake1, snake2, tomatoes)

    count = 1
    i = 0
    input1, input2 = Direction.RIGHT, Direction.LEFT
    prev = defaultdict(bool)

    run = True
    while run:
        key = pygame.key.get_pressed()

        # Handle input 60 times a second
        if count < FRAMERATE // TICKS_PER_S:
            if not load_game:
                input1, input2, prev = handle_input(key, prev, input1, input2)
            count += 1
        else:
            # Update game state based on player input
            count = 1
            if load_game:
                input1, input2 = snake1_actions[i], snake2_actions[i]
                i += 1
            if load_game:
                grid, snake1, snake2, tomatoes, _, _, tomato_positions = handle_movement(game_state, input1, input2, config, tomato_positions=tomato_positions)
            else:
                grid, snake1, snake2, tomatoes, _ = handle_movement(game_state, input1, input2, config)
            grid, snake1, snake2, _ = game_state
            if (not snake1.alive) and (not snake2.alive):
                run = False
            # if (not snake1.alive and snake1.length < snake2.length) or (not snake2.alive and snake1.length > snake2.length): FIXME: UNCOMMENT
            #     run = False

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
        text_surface = length_font.render(f'Green Snake: {snake1.length}     Blue Snake: {snake2.length}', False, (25, 136, 191))
        screen.blit(text_surface, (40, 0))

        # Event handler, currently just for closing game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        
        # Refreshes display
        pygame.display.update()

        # Sets frame rate
        clock.tick(FRAMERATE)

    if snake1.length > snake2.length:
        end_text = "Green Snake Wins!"
    elif snake2.length > snake1.length:
        end_text = "Blue Snake Wins!"
    else:
        end_text = "It's a Tie!"

    length_font = pygame.font.SysFont('ubuntusans', 75)
    text_surface = length_font.render(end_text, False, WHITE)
    screen.blit(text_surface, (200, 450))
    pygame.display.update()

    pygame.time.wait(1500)
    pygame.quit()

def replay_game(load_path):
    # Load from file
    with open(load_path, "rb") as f:
        snake1_actions, snake2_actions, tomato_positions = pickle.load(f)

    saved_game = (snake1_actions, snake2_actions, tomato_positions)
    two_player_game(saved_game)

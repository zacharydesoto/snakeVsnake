import pygame

from environment import SnakeEnvironment
from utils import *
from movement import *

def game(ticks_per_s=5, snake1_model=None, snake2_model=None):
    pygame.init()

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
    TICKS_PER_S = ticks_per_s

    screen = pygame.display.set_mode((SCREEN_WIDTH+2*MARGIN, SCREEN_HEIGHT+2*MARGIN + TOP_MARGIN))
    clock = pygame.time.Clock()

    pygame.display.set_caption('SnakeRL')
    pygame.font.init()

    count = 1
    input1, input2 = Direction.RIGHT, Direction.LEFT
    prev = defaultdict(bool)

    env = SnakeEnvironment(config)

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
            if snake1_model:
                if snake1_model.baseline:
                    input1 = snake1_model.get_action(env.get_portion_grid(is_snake1=True), train=False, env=env)
                else:
                    input1 = snake1_model.get_action(env.get_portion_grid(is_snake1=True), train=False)
            if snake2_model:
                if snake2_model.baseline:
                    input2 = snake2_model.get_action(env.get_portion_grid(is_snake1=False), train=False, env=env)
                else:
                    input2 = snake2_model.get_action(env.get_portion_grid(is_snake1=False), train=False)
            _, _, _, terminated, truncated = env.step(input1, input2)
            if terminated or truncated:
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

        grid = env.get_grid()
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

        length_font = pygame.font.SysFont('ubuntusans', 40)
        text_surface = length_font.render(f'Green Snake: {env.snake1.length}     Blue Snake: {env.snake2.length}', False, (25, 136, 191))
        screen.blit(text_surface, (40, 0))

        # Event handler, currently just for closing game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        
        # Refreshes display
        pygame.display.update()

        # Sets frame rate
        clock.tick(FRAMERATE)

    if env.snake1.length > env.snake2.length:
        end_text = "Green Snake Wins!"
    elif env.snake2.length > env.snake1.length:
        end_text = "Blue Snake Wins!"
    else:
        end_text = "It's a Tie!"

    length_font = pygame.font.SysFont('ubuntusans', 40)
    text_surface = length_font.render(end_text, False, WHITE)
    screen.blit(text_surface, (200, 450))
    pygame.display.update()

    pygame.time.wait(1500)
    pygame.quit()
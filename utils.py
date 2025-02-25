def grid_to_screen(grid_x, grid_y, config):
    screen_x = grid_x * config['SCREEN_WIDTH'] / config['GRID_WIDTH']
    screen_y = grid_y * config['SCREEN_HEIGHT'] / config['GRID_HEIGHT']
    return screen_x, screen_y
import pygame

pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

player1 = pygame.Rect((300, 250, 50, 50))

run = True
while run:

    screen.fill((0, 0, 0))

    pygame.draw.rect(screen, (255, 0, 0), player1)

    key = pygame.key.get_pressed()
    if key[pygame.K_a]:
        player1.move_ip(-1, 0)
    elif key[pygame.K_d]:
        player1.move_ip(1, 0)
    elif key[pygame.K_w]:
        player1.move_ip(0, -1)
    elif key[pygame.K_s]:
        player1.move_ip(0, 1)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    
    pygame.display.update()

pygame.quit()
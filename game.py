# Example file showing a basic pygame "game loop"
import pygame
# pong game
# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True

player1 = pygame.Rect(10, 10, 10, 100)
player2 = pygame.Rect(1260, 10, 10, 100)
ball = pygame.Rect(640, 360, 10, 10)

ball_speed_x = 5
ball_speed_y = 5

autopilot = True
autopilot_speed = 5

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("black")

    # RENDER YOUR GAME HERE
    pygame.draw.rect(screen, "white", player1)
    pygame.draw.rect(screen, "white", player2)
    pygame.draw.rect(screen, "white", ball)

    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        player1.y -= 5
    if keys[pygame.K_s]:
        player1.y += 5
    if keys[pygame.K_UP]:
        player2.y -= 5
    if keys[pygame.K_DOWN]:
        player2.y += 5

    if autopilot:
        if player2.y > ball.y:
            player2.y -= autopilot_speed
        if player2.y < ball.y:
            player2.y += autopilot_speed

    ball.x += ball_speed_x
    ball.y += ball_speed_y

    if ball.y <= 0 or ball.y >= 710:
        ball_speed_y *= -1
    if ball.colliderect(player1) or ball.colliderect(player2):
        ball_speed_x *= -1

    if ball.x <= 0 or ball.x >= 1270:
        ball.x = 640
        ball.y = 360
        ball_speed_x *= -1
        ball_speed_y *= -1


    # flip() the display to put your work on screen
    pygame.display.flip()

    clock.tick(60)  # limits FPS to 60

pygame.quit()
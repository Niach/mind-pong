from train import generate_data
from time import sleep
from threading import Thread
import pygame
from multiprocessing import Process


# pong game
# pygame setup





def capture_process():
    global running
    while running:
        data = next(generate_data())
        print(data)



if __name__ == "__main__":
    pygame.init()
    # screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
    screen = pygame.display.set_mode((1280, 720))
    clock = pygame.time.Clock()
    running = True
    booly = True

    p = Process(target=capture_process)
    p.start()


    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # fill the screen with a color to wipe away anything from last frame
        if (booly):
            screen.fill("black")
            booly = False
        else:
            screen.fill("white")
            booly = True

        # RENDER YOUR GAME HERE
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False

        # update the display

        pygame.display.flip()
        clock.tick(1)
    pygame.quit()
    p.kill()
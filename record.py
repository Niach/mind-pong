import multiprocessing

from train import generate_data
from time import sleep
from threading import Thread
import pygame
from multiprocessing import Process, Pipe
import time


# pong game
# pygame setup





def capture_process(event):
    running = True
    start_time = time.time_ns()
    datas = []
    while running:
        datas.append(next(generate_data()))
        if time.time_ns() - start_time > 1000000000:
            event.set()
            start_time = time.time_ns()
            print(len(datas))
            datas = []



if __name__ == "__main__":

    process_event = multiprocessing.Event()

    p = Process(target=capture_process, args=(process_event,))
    p.start()

    pygame.init()
    # screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
    screen = pygame.display.set_mode((1280, 720))
    clock = pygame.time.Clock()
    running = True
    booly = True




    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # fill the screen with a color to wipe away anything from last frame
        if (booly):
            screen.fill("black")
        else:
            screen.fill("white")

        # RENDER YOUR GAME HERE
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False

        # update the display
        pygame.display.flip()
        clock.tick(60)
        if process_event.is_set():
            booly = not booly
            process_event.clear()





    pygame.quit()
    p.kill()
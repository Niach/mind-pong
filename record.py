import dataclasses
import json
import multiprocessing
import os
import random
import socket
import time
from multiprocessing import Process

import numpy as np
import pygame

from models import EEGSample, EEGData


def capture_process(left_event, right_event, rest_event):
    ip = "127.0.0.1"
    port = 1000

    listening_address = (ip, port)

    dg_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dg_socket.bind(listening_address)

    running = True
    recording_states = ["LEFT", "RIGHT", "REST"]
    current_state = "REST"

    create_class_directories(recording_states)

    rest_event.set()

    start_time = time.time_ns()
    datas = np.empty(1)
    is_first_sample = True

    while running:
        (data, _) = dg_socket.recvfrom(1024)
        number_of_bytes_received = len(data)

        if number_of_bytes_received > 0:
            message_byte = np.frombuffer(data, dtype=np.uint8, count=number_of_bytes_received)
            message = message_byte.tobytes().decode('ascii')
            data_list: list[float] = [float(item) for item in message.split(',')]
            data_array: np.ndarray[EEGData] = np.array(data_list)
            np.append(datas, data_array)

        if time.time_ns() - start_time > 10000000000:
            sample = EEGSample(datas, start_time, time.time_ns(), current_state)
            current_state = random.choice(recording_states)
            datas = np.empty(1)

            if current_state == "LEFT":
                left_event.set()
            elif current_state == "RIGHT":
                right_event.set()
            elif current_state == "REST":
                rest_event.set()

            if not is_first_sample:
                # save sample as numpy array
                with open('data/raw/' + current_state + '/' + str(time.time_ns()) + '.json', 'w', encoding='UTF8') as f:
                    f.write(json.dumps(dataclasses.asdict(sample), default=lambda o: dataclasses.asdict(o), indent=4))

            start_time = time.time_ns()


def create_class_directories(classes):
    for class_ in classes:
        raw = os.path.join("data", "raw", class_)
        if not os.path.exists(raw):
            os.makedirs(raw)


WIDTH = 1920
HEIGHT = 1080


def draw_arrow_left(screen):
    pygame.draw.polygon(screen, "white",
                        [(WIDTH * 0.5 - WIDTH * 0.039, HEIGHT * 0.139), (WIDTH * 0.5 - WIDTH * 0.039, HEIGHT * 0.278),
                         (WIDTH * 0.5 - WIDTH * 0.117, HEIGHT * 0.208)])
    pygame.draw.rect(screen, "white",
                     pygame.Rect(WIDTH * 0.5 - WIDTH * 0.039, HEIGHT * 0.187, WIDTH * 0.078, HEIGHT * 0.042))


def draw_arrow_right(screen):
    pygame.draw.polygon(screen, "white",
                        [(WIDTH * 0.5 + WIDTH * 0.039, HEIGHT * 0.139), (WIDTH * 0.5 + WIDTH * 0.039, HEIGHT * 0.278),
                         (WIDTH * 0.5 + WIDTH * 0.117, HEIGHT * 0.208)])
    pygame.draw.rect(screen, "white",
                     pygame.Rect(WIDTH * 0.5 - WIDTH * 0.039, HEIGHT * 0.187, WIDTH * 0.078, HEIGHT * 0.042))


def draw_rest(screen):
    pygame.draw.circle(screen, "white", (WIDTH * 0.5, HEIGHT * 0.208), WIDTH * 0.039)
    pygame.draw.circle(screen, "black", (WIDTH * 0.5, HEIGHT * 0.208), WIDTH * 0.023)


if __name__ == "__main__":

    left_event = multiprocessing.Event()
    right_event = multiprocessing.Event()
    rest_event = multiprocessing.Event()

    p = Process(target=capture_process, args=(left_event, right_event, rest_event))
    p.start()

    pygame.init()
    screen = pygame.display.set_mode((1920, 1080))
    clock = pygame.time.Clock()
    running = True

    current_direction = "REST"
    current_x = WIDTH * 0.5 - WIDTH * 0.039

    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # fill the screen with a color to wipe away anything from last frame

        screen.fill("black")

        # RENDER YOUR GAME HERE
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False

        if left_event.is_set():
            current_direction = "LEFT"
            current_x = WIDTH * 0.805
            left_event.clear()

        if right_event.is_set():
            current_direction = "RIGHT"
            current_x = WIDTH * 0.117
            right_event.clear()

        if rest_event.is_set():
            current_direction = "REST"
            current_x = WIDTH * 0.5 - WIDTH * 0.039
            rest_event.clear()

        if current_direction == "LEFT":
            current_x = current_x - 0.5
            draw_arrow_left(screen)

        elif current_direction == "RIGHT":
            current_x = current_x + 0.5
            draw_arrow_right(screen)

        elif current_direction == "REST":
            draw_rest(screen)

        pygame.draw.rect(screen, "white", pygame.Rect(current_x, HEIGHT * 0.833, WIDTH * 0.078, HEIGHT * 0.02))

        # update the display
        pygame.display.flip()
        clock.tick(144)

    pygame.quit()
    p.kill()

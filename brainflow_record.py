import argparse
import multiprocessing
import os
import time
from multiprocessing import Process
from pprint import pprint

import numpy as np
import pygame
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

import random

WIDTH = 1280
HEIGHT = 1080

PLAYER_WIDTH_MOD = 0.2
PLAYER_HEIGHT_MOD = 0.02

BALL_WIDTH_MOD = 0.02
BALL_HEIGHT_MOD = 0.02


def start_capture():
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    board = BoardShim(BoardIds.UNICORN_BOARD, params)
    board.prepare_session()
    pprint(BoardShim.get_board_descr(BoardIds.UNICORN_BOARD))

    board.start_stream()

    board.insert_marker(1)

    return board


def create_class_directories(classes):
    for class_ in classes:
        raw = os.path.join("data", "raw", class_)
        if not os.path.exists(raw):
            os.makedirs(raw)


class Player(pygame.Rect):

    def __init__(self, x, y):
        super().__init__(x, y, WIDTH * PLAYER_WIDTH_MOD, HEIGHT * PLAYER_HEIGHT_MOD)
        self.vx = 0
        self.beep_left = Beep(True)
        self.beep_right = Beep(False)
        self.last_left_pressed = False
        self.last_right_pressed = False

    def render(self, screen):
        pygame.draw.rect(screen, "white",
                         pygame.Rect(self.x, self.y, self.width, self.height))

    def calc(self, keys, dt, board):

        if keys[pygame.K_LCTRL]:
            self.vx = -150
            self.beep_right.stop()
            self.beep_left.play()
            self.last_right_pressed = False

            if not self.last_left_pressed:
                board.insert_marker(1)
                self.last_left_pressed = True
                print("LEFT")

        elif keys[pygame.K_RCTRL]:
            self.vx = 150
            self.beep_left.stop()
            self.beep_right.play()
            self.last_left_pressed = False
            if not self.last_right_pressed:
                board.insert_marker(2)
                self.last_right_pressed = True
                print("RIGHT")


        else:
            self.vx = 0
            self.beep_left.stop()
            self.beep_right.stop()

            if self.last_left_pressed or self.last_right_pressed:
                board.insert_marker(3)
                print("CLEAR")


            self.last_left_pressed = False
            self.last_right_pressed = False

        self.x += self.vx * dt


class Ball(pygame.Rect):

    def __init__(self, x, y):
        super().__init__(x, y, WIDTH * BALL_WIDTH_MOD, HEIGHT * BALL_HEIGHT_MOD)
        self.vx = 0
        self.vy = 125

    def render(self, screen):
        pygame.draw.rect(screen, "white", self)

    def update(self, player, boxes, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt

        if self.x <= 0 or self.x >= WIDTH:
            self.vx = -self.vx

        if self.y <= 0 or self.y >= HEIGHT:
            self.vy = -self.vy

        if self.colliderect(player):
            self.vy = -self.vy
            self.vx = self.vx + random.randint(-75, 75)
            if self.vx > 250:
                self.vx = 250
            elif self.vx < -250:
                self.vx = -250

        for box in boxes:
            if self.colliderect(box):
                self.vy = -self.vy
                boxes.remove(box)
                break


class Box(pygame.Rect):
    def __init__(self, x, y):
        super().__init__(x, y, WIDTH / 11, HEIGHT * 2 * BALL_HEIGHT_MOD)

    def render(self, screen):
        pygame.draw.rect(screen, "white", self)


class Beep:

    def __init__(self, is_left):
        # Constants
        FS = 44100  # Sampling rate
        T = 5  # Duration in seconds
        FREQ = 100 if is_left else 110 # Frequency in Hz

        # Generate the sine wave samples for the left channel
        t = np.linspace(0, T, int(FS * T), endpoint=False)  # Time array
        sin_channel = np.sin(2 * np.pi * FREQ * t)

        if is_left:
            left_channel_pcm = (32767 * sin_channel).astype(np.int16)
            right_channel_pcm = np.zeros(left_channel_pcm.shape, dtype=np.int16)
        else:
            right_channel_pcm = (32767 * sin_channel).astype(np.int16)
            left_channel_pcm = np.zeros(right_channel_pcm.shape, dtype=np.int16)

        # Combine both channels
        stereo_pcm = np.column_stack((left_channel_pcm, right_channel_pcm))

        pygame.mixer.init(FS, -16, 2)
        self.sound = pygame.sndarray.make_sound(stereo_pcm)

    def play(self):
        if not pygame.mixer.get_busy():
            self.sound.play()

    def stop(self):
        self.sound.stop()



def save_sample(board):
    data = board.get_board_data()
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.UNICORN_BOARD)  # 0-8
    marker_channel = BoardShim.get_marker_channel(BoardIds.UNICORN_BOARD)
    eeg_data = data[eeg_channels, :]
    eeg_data = np.append(eeg_data, [data[marker_channel, :]], axis=0)
    eeg_np = np.asarray(eeg_data)

    np.savetxt('data/raw/' + str(time.time_ns()) + '.csv', eeg_np, delimiter=",")

if __name__ == "__main__":
    create_class_directories(["LEFT", "RIGHT", "REST"])

    board = start_capture()
    running = True
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    dt = 0
    player = Player(WIDTH * 0.5 - (0.5 * PLAYER_WIDTH_MOD * WIDTH), HEIGHT - (2 * PLAYER_HEIGHT_MOD * HEIGHT))
    ball = Ball(WIDTH * 0.5 - (0.5 * BALL_WIDTH_MOD * WIDTH), HEIGHT * 0.5 - (2 * BALL_HEIGHT_MOD * HEIGHT))
    boxes = []

    current_time = time.time()

    for i in range(10):
        for j in range(4):
            boxes.append(Box((WIDTH / 11 / 10 / 2) + WIDTH * 0.1 * i,
                             ((2 * BALL_HEIGHT_MOD * HEIGHT) * j + j * (BALL_HEIGHT_MOD * HEIGHT / 2)) + (
                                         BALL_HEIGHT_MOD * HEIGHT / 2)))

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill("black")
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False

        ball.update(player, boxes, dt)
        player.calc(keys, dt, board)

        player.render(screen)
        ball.render(screen)
        for box in boxes:
            box.render(screen)

        pygame.display.flip()
        if time.time() - current_time > 10:
            current_time = time.time()
            save_sample(board)

        dt = clock.tick(60) / 1000

    board.stop_stream()
    board.release_session()
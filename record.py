import dataclasses
import json
import multiprocessing
import os
import random
import socket
import time
from multiprocessing import Process
import pickle

import numpy as np
import pygame

from game import PongGame
from models import EEGSample, EEGData


def capture_process(left_event, right_event, rest_event):
    ip = "127.0.0.1"
    port = 1000

    listening_address = (ip, port)

    dg_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dg_socket.bind(listening_address)

    running = True
    recording_states = ["LEFT", "RIGHT", "REST", "REST", "REST"]
    current_state = "REST"

    create_class_directories(recording_states)

    rest_event.set()

    start_time = time.time_ns()
    datas = []
    is_first_sample = True

    while running:
        (data, _) = dg_socket.recvfrom(1024)
        number_of_bytes_received = len(data)

        if number_of_bytes_received > 0:
            message_byte = np.frombuffer(data, dtype=np.uint8, count=number_of_bytes_received)
            message = message_byte.tobytes().decode('ascii')
            data_list: list[float] = [float(item) for item in message.split(',')]
            eeg_data = EEGData(channelData=data_list, timestamp=time.time_ns())
            datas.append(eeg_data)

        if time.time_ns() - start_time > 5000000000:
            sample = EEGSample(datas, start_time, time.time_ns(), current_state)
            current_state = random.choice(recording_states)
            datas = []

            if current_state == "LEFT":
                left_event.set()
            elif current_state == "RIGHT":
                right_event.set()
            elif current_state == "REST":
                rest_event.set()

            if not is_first_sample:
                with open('data/raw/' + current_state + '/' + str(time.time_ns()) + '.pickle', 'wb') as f:
                    pickle.dump(sample, f, pickle.HIGHEST_PROTOCOL)

            else:
                is_first_sample = False

            start_time = time.time_ns()


def create_class_directories(classes):
    for class_ in classes:
        raw = os.path.join("data", "raw", class_)
        if not os.path.exists(raw):
            os.makedirs(raw)





if __name__ == "__main__":
    left_event = multiprocessing.Event()
    right_event = multiprocessing.Event()
    rest_event = multiprocessing.Event()

    p = Process(target=capture_process, args=(left_event, right_event, rest_event))
    p.start()

    game = PongGame(left_event, right_event, rest_event)
    game.run()

    p.kill()

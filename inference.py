import multiprocessing
import random
import socket
import time
from multiprocessing import Process

import numpy as np
import torch
import pygame

from game import PongGame
from models import classes, EEGNet


def predict(model, inputs):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted_indices = torch.max(outputs.data, 1)
    return [classes[idx] for idx in predicted_indices]


def load_model():
    print('loading model')
    model = EEGNet()  # Instantiate the empty model
    model.load_state_dict(torch.load('model.pt'))  # Load the state_dict
    model.eval()  # Set it to evaluation mode
    return model


def capture_process(left_event, right_event, rest_event):
    model = load_model()

    print('binding to udp')

    ip = "127.0.0.1"
    port = 1000

    listening_address = (ip, port)

    dg_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dg_socket.bind(listening_address)

    running = True

    current_state = "REST"

    rest_event.set()

    start_time = time.time_ns()
    datas = []

    while running:
        pass
        # (data, _) = dg_socket.recvfrom(1024)
        # number_of_bytes_received = len(data)
        #
        # if number_of_bytes_received > 0:
        #     message_byte = np.frombuffer(data, dtype=np.uint8, count=number_of_bytes_received)
        #     message = message_byte.tobytes().decode('ascii')
        #     data_list: list[float] = [float(item) for item in message.split(',')]
        #     eeg_data = EEGData(channelData=data_list, timestamp=time.time_ns())
        #     datas.append(eeg_data)
        #
        # if time.time_ns() - start_time > 5000000000:
        #     sample = EEGSample(datas, start_time, time.time_ns(), current_state)
        #
        #     input = torch.tensor(map_sample(sample)).float().unsqueeze(0)
        #     current_state = predict(model, input)[0]
        #     print("predicted state:", current_state)
        #
        #     if current_state == "LEFT":
        #         left_event.set()
        #     elif current_state == "RIGHT":
        #         right_event.set()
        #     elif current_state == "REST":
        #         rest_event.set()
        #
        #     start_time = time.time_ns()


if __name__ == "__main__":

    left_event = multiprocessing.Event()
    right_event = multiprocessing.Event()
    rest_event = multiprocessing.Event()

    p = Process(target=capture_process, args=(left_event, right_event, rest_event))
    p.start()

    game = PongGame(left_event, right_event, rest_event)
    game.run()

    p.kill()


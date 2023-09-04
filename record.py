from dataclasses import dataclass
from math import e
import multiprocessing
import numpy as np
from train import generate_data
from time import sleep
from threading import Thread
import pygame
from multiprocessing import Process, Pipe
import time
import socket
import random



@dataclass
class EEGData:
    channelData: list
    timestamp: int
    
@dataclass
class EEGSample:
    data: EEGData
    timestamp_start: int
    timestamp_end: int
    state: str
    

def capture_process(left_event, right_event, rest_event):
    ip = "127.0.0.1"
    port = 1000
    # create out dir and save shit

    listeningAddress = (ip, port)

    dgSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dgSocket.bind(listeningAddress)    

    running = True
    recording_states = ["LEFT", "RIGHT", "REST"]
    current_state = "REST"
    
    left_event.set()



    start_time = time.time_ns()
    datas = []
    while running:
        (data, _) = dgSocket.recvfrom(1024)
        numberOfBytesReceived = len(data)
    
        if numberOfBytesReceived > 0:
            messageByte = np.frombuffer(data, dtype=np.uint8, count=numberOfBytesReceived)
            message = messageByte.tobytes().decode('ascii')
            data_list = [float(item) for item in message.split(',')]
            data_array = np.array(data_list)       
            datas.append(EEGData(data_array, time.time_ns()))





        if time.time_ns() - start_time > 10000000000:
            sample = EEGSample(datas, start_time, time.time_ns, current_state)            
            current_state = random.choice(recording_states)            
            datas = []
               
            if current_state == "LEFT":
                left_event.set()
            elif current_state == "RIGHT":
                right_event.set()
            elif current_state == "REST":
                rest_event.set()

            start_time = time.time_ns()
            
            





if __name__ == "__main__":

    left_event = multiprocessing.Event()
    right_event = multiprocessing.Event()
    rest_event = multiprocessing.Event()

    p = Process(target=capture_process, args=(left_event, right_event, rest_event))
    p.start()

    pygame.init()
    # screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
    screen = pygame.display.set_mode((1280, 720))
    clock = pygame.time.Clock()
    running = True
    
    current_direction = "REST"
    current_x = 1280 / 2 - 50


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
            current_x = 1030
            left_event.clear()

        if right_event.is_set():
            current_direction = "RIGHT"
            current_x = 150
            right_event.clear()

        if rest_event.is_set():
            current_direction = "REST"
            current_x = 1280 / 2 - 50
            rest_event.clear()


        if current_direction == "LEFT":
            current_x = current_x - 0.5

        elif current_direction == "RIGHT":
            current_x = current_x + 0.5
            

        pygame.draw.rect(screen, "white", pygame.Rect(current_x, 600, 100, 10))

        # update the display
        pygame.display.flip()
        clock.tick(144)



    pygame.quit()
    p.kill()
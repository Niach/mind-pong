import argparse
import multiprocessing
import os
import time
from multiprocessing import Process
from pprint import pprint

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets

from game import PongGame



def capture_process(left_event, right_event, rest_event):
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    board = BoardShim(BoardIds.UNICORN_BOARD, params)
    board.prepare_session()
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.UNICORN_BOARD) #0-8
    pprint(BoardShim.get_board_descr(BoardIds.UNICORN_BOARD))
    running = True
    recording_states = ["LEFT", "RIGHT", "REST", "REST", "REST"]
    current_state = "REST"

    create_class_directories(recording_states)

    rest_event.set()

    start_time = time.time_ns()
    is_first_sample = True

    board.start_stream()
    while running:
        time.sleep(1)
        data = board.get_board_data()
        eeg_data = data[eeg_channels, :]
        eeg_np = np.asarray(eeg_data)
        np.savetxt('data/raw/' + current_state + '/' + str(time.time_ns()) + '.csv', eeg_np, delimiter=",")
        print(len(eeg_data))



        pass


        # if number_of_bytes_received > 0:
        #     message_byte = np.frombuffer(data, dtype=np.uint8, count=number_of_bytes_received)
        #     message = message_byte.tobytes().decode('ascii')
        #     data_list: list[float] = [float(item) for item in message.split(',')]
        #     eeg_data = EEGData(channelData=data_list, timestamp=time.time_ns())
        #     datas.append(eeg_data)
        #
        # if time.time_ns() - start_time > 5000000000:
        #     sample = EEGSample(datas, start_time, time.time_ns(), current_state)
        #     current_state = random.choice(recording_states)
        #     datas = []
        #
        #     if current_state == "LEFT":
        #         left_event.set()
        #     elif current_state == "RIGHT":
        #         right_event.set()
        #     elif current_state == "REST":
        #         rest_event.set()
        #
        #     if not is_first_sample:
        #         with open('data/raw/' + current_state + '/' + str(time.time_ns()) + '.pickle', 'wb') as f:
        #             pickle.dump(sample, f, pickle.HIGHEST_PROTOCOL)
        #
        #     else:
        #         is_first_sample = False
        #
        #     start_time = time.time_ns()


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

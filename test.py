import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

import mne
from mne.channels import read_layout


def main():
    params = BrainFlowInputParams()
    board = BoardShim(BoardIds.UNICORN_BOARD, params)
    board.prepare_session()
    board.start_stream()
    time.sleep(10)
    data = board.get_board_data()
    board.stop_stream()
    board.release_session()
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.UNICORN_BOARD.value)
    eeg_data = data[eeg_channels, :]
    eeg_data = eeg_data / 1000000 # BrainFlow returns uV, convert to V for MNE

    ch_types = ['eeg'] * len(eeg_channels)
    ch_names = BoardShim.get_eeg_names(BoardIds.UNICORN_BOARD.value)
    sfreq = BoardShim.get_sampling_rate(BoardIds.UNICORN_BOARD.value)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_data, info)
    # its time to plot something!
    raw.compute_psd().plot()

if __name__ == "__main__":
    main()

from dataclasses import dataclass

import numpy as np


@dataclass
class EEGData:
    channelData: np.ndarray[float]
    timestamp: int


@dataclass
class EEGSample:
    data: np.ndarray[EEGData]
    timestamp_start: int
    timestamp_end: int
    state: str

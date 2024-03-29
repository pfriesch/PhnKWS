from enum import Enum


class DatasetType(Enum):
    FRAMEWISE_SHUFFLED_FRAMES = 1
    FRAMEWISE_SEQUENTIAL = 2
    FRAMEWISE_SEQUENTIAL_APPENDED_CONTEXT = 3
    SEQUENTIAL = 4
    SEQUENTIAL_APPENDED_CONTEXT = 5

"""
Multi-rate turbo-code configuration.
"""

import numpy as np

RANDOM_SEED = 12
INFORMATION_BLOCK_LENGTH = 4000

SUPPORTED_CODE_RATES = {
    "1/3": 1.0 / 3.0,
    "1/2": 1.0 / 2.0,
    "3/4": 3.0 / 4.0,
    "7/8": 7.0 / 8.0,
}

SELECTED_CODE_RATE = SUPPORTED_CODE_RATES["1/3"]
DECODER_ITERATION_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

CONVOLUTIONAL_EB_NO_DB = np.arange(-4.0, 5.0, 1.0, dtype=float)
TURBO_EB_NO_DB = np.arange(-1.0, 1.76, 0.25, dtype=float)

DEBUG_MODE = True
CONVOLUTIONAL_MINIMUM_FRAME_COUNT = 80 if DEBUG_MODE else 200
CONVOLUTIONAL_MAXIMUM_FRAME_COUNT = 400 if DEBUG_MODE else 1200
CONVOLUTIONAL_TARGET_ERROR_COUNT = 400 if DEBUG_MODE else 1200

TURBO_MINIMUM_FRAME_COUNT = 80 if DEBUG_MODE else 180
TURBO_MAXIMUM_FRAME_COUNT = 500 if DEBUG_MODE else 1400
TURBO_TARGET_ERROR_COUNT = 150 if DEBUG_MODE else 500

BENCHMARK_BLOCK_COUNT = 25 if DEBUG_MODE else 80
BENCHMARK_EB_NO_DB = 0.5

SAVE_FIGURES = False
SAVE_PREFIX = "turbo_multirate"

RSC_MEMORY = 2
RSC_TAIL_LENGTH = 2
RSC_STATE_COUNT = 2 ** RSC_MEMORY

CONVOLUTIONAL_MINIMUM_FRAMES = CONVOLUTIONAL_MINIMUM_FRAME_COUNT
CONVOLUTIONAL_MAXIMUM_FRAMES = CONVOLUTIONAL_MAXIMUM_FRAME_COUNT
CONVOLUTIONAL_TARGET_ERRORS = CONVOLUTIONAL_TARGET_ERROR_COUNT
TURBO_MINIMUM_FRAMES = TURBO_MINIMUM_FRAME_COUNT
TURBO_MAXIMUM_FRAMES = TURBO_MAXIMUM_FRAME_COUNT
TURBO_TARGET_ERRORS = TURBO_TARGET_ERROR_COUNT

def get_rate_label():
    for label, value in SUPPORTED_CODE_RATES.items():
        if abs(value - SELECTED_CODE_RATE) < 1e-12:
            return label
    return f"{SELECTED_CODE_RATE:.3f}"

def get_information_puncture_definition():
    """
    Return puncturing patterns for the information-bit section only.
    Tail parity symbols are always transmitted.
    """
    rate = SELECTED_CODE_RATE

    if abs(rate - 1.0 / 3.0) < 1e-12:
        return {
            "parity_1_pattern": np.array([1], dtype=np.int8),
            "parity_2_pattern": np.array([1], dtype=np.int8),
        }

    if abs(rate - 1.0 / 2.0) < 1e-12:
        return {
            "parity_1_pattern": np.array([1, 0], dtype=np.int8),
            "parity_2_pattern": np.array([0, 1], dtype=np.int8),
        }

    if abs(rate - 3.0 / 4.0) < 1e-12:
        return {
            "parity_1_pattern": np.array([1, 0, 0, 0], dtype=np.int8),
            "parity_2_pattern": np.array([0, 1, 0, 0], dtype=np.int8),
        }

    if abs(rate - 7.0 / 8.0) < 1e-12:
        return {
            "parity_1_pattern": np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8),
            "parity_2_pattern": np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int8),
        }

    raise ValueError(f"Unsupported turbo code rate: {rate}")

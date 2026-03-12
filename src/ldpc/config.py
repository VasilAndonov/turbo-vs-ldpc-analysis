"""
Multi-rate LDPC configuration.
"""

import numpy as np

RANDOM_SEED = 12
INFORMATION_BIT_COUNT = 256

SUPPORTED_CODE_RATES = {
    "1/2": 1.0 / 2.0,
    "1/3": 1.0 / 3.0,
    "3/4": 3.0 / 4.0,
    "7/8": 7.0 / 8.0,
}

SELECTED_CODE_RATE = SUPPORTED_CODE_RATES["1/3"]

CODEWORD_BIT_COUNT = int(round(INFORMATION_BIT_COUNT / SELECTED_CODE_RATE))
PARITY_BIT_COUNT = CODEWORD_BIT_COUNT - INFORMATION_BIT_COUNT

INFORMATION_COLUMN_WEIGHT_BY_RATE = {
    "1/2": 5,
    "1/3": 6,
    "3/4": 4,
    "7/8": 3,
}

ROW_WEIGHT_PENALTY = 0.35
DECODER_ITERATION_LIST = [1, 2, 3, 4, 5, 6]
NORMALIZATION_FACTOR = 0.80

DEBUG_MODE = True
LDPC_EB_NO_DB = np.arange(-1.0, 1.76, 0.25, dtype=float)
MINIMUM_FRAME_COUNT = 80 if DEBUG_MODE else 200
MAXIMUM_FRAME_COUNT = 600 if DEBUG_MODE else 1500
TARGET_ERROR_COUNT = 160 if DEBUG_MODE else 500

SAVE_FIGURES = False
SAVE_PREFIX = "ldpc_multirate"

def get_rate_label():
    for label, value in SUPPORTED_CODE_RATES.items():
        if abs(value - SELECTED_CODE_RATE) < 1e-12:
            return label
    return f"{SELECTED_CODE_RATE:.3f}"

def get_information_column_weight():
    return INFORMATION_COLUMN_WEIGHT_BY_RATE[get_rate_label()]


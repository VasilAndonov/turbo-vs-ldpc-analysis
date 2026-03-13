import numpy as np

RANDOM_SEED = 12
INFORMATION_BIT_COUNT = 1024

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
    "1/2": 6,
    "1/3": 7,
    "3/4": 5,
    "7/8": 4,
}

PARITY_BAND_WIDTH_BY_RATE = {
    "1/2": 5,
    "1/3": 6,
    "3/4": 4,
    "7/8": 3,
}

ROW_WEIGHT_PENALTY = 0.55

# Give LDPC a larger practical iteration budget.
DECODER_ITERATION_LIST = [1, 2, 4, 6, 8, 10]

LLR_CLIP = 20.0
MESSAGE_DAMPING = 0.20

DEBUG_MODE = False
LDPC_EB_NO_DB = np.arange(-1.0, 1.76, 0.25, dtype=float)

MINIMUM_FRAME_COUNT = 160 if DEBUG_MODE else 320
MAXIMUM_FRAME_COUNT = 1000 if DEBUG_MODE else 2200
TARGET_ERROR_COUNT = 320 if DEBUG_MODE else 1000

BENCHMARK_BLOCK_COUNT = 16 if DEBUG_MODE else 24
BENCHMARK_EB_NO_DB = 0.5

SAVE_FIGURES = False
SAVE_PREFIX = "ldpc_improved"


def get_rate_label():
    for label, value in SUPPORTED_CODE_RATES.items():
        if abs(value - SELECTED_CODE_RATE) < 1e-12:
            return label
    return f"{SELECTED_CODE_RATE:.3f}"


def get_information_column_weight():
    return INFORMATION_COLUMN_WEIGHT_BY_RATE[get_rate_label()]


def get_parity_band_width():
    return PARITY_BAND_WIDTH_BY_RATE[get_rate_label()]

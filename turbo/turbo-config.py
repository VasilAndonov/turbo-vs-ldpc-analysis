import numpy as np

FAST_MODE = True
SHOW_PLOTS = True
SAVE_PLOTS = False
PLOT_PREFIX = "turbo"

RANDOM_SEED = 12
INFORMATION_BITS = 256 if FAST_MODE else 1024
ITERATIONS = [1, 2, 3, 4, 5, 6]

CONV_EBN0_DB = np.array([-8.5, -6.8, -4.8, -2.8, -0.8, 1.5, 3.5], dtype=float)
TURBO_EBN0_DB = np.array([-1.0, 0.0, 0.8, 1.0, 1.15, 1.25, 1.3], dtype=float) if FAST_MODE else np.arange(-1.0, 1.31, 0.15)

MIN_FRAMES = 20 if FAST_MODE else 120
MAX_FRAMES = 70 if FAST_MODE else 400
TARGET_ERRORS = 80 if FAST_MODE else 300
BENCHMARK_BLOCKS = 8 if FAST_MODE else 20

RSC_MEMORY = 2
RSC_TAIL_LENGTH = 2
RSC_STATE_COUNT = 2 ** RSC_MEMORY

SUPPORTED_CODE_RATES = {"1/3": 1/3, "1/2": 1/2, "3/4": 3/4, "7/8": 7/8}
SELECTED_CODE_RATE_LABEL = "1/3"

def sigma2_from_ebn0(ebn0_db, code_rate):
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    return 1.0 / (2.0 * code_rate * ebn0)

def get_puncture_definition(rate_label):
    if rate_label == "1/3":
        return np.array([1], dtype=np.int8), np.array([1], dtype=np.int8)
    if rate_label == "1/2":
        return np.array([1, 0], dtype=np.int8), np.array([0, 1], dtype=np.int8)
    if rate_label == "3/4":
        return np.array([1, 0, 0, 1], dtype=np.int8), np.array([0, 1, 1, 0], dtype=np.int8)
    if rate_label == "7/8":
        return np.array([1, 0, 0, 0, 0, 1, 0, 0], dtype=np.int8), np.array([0, 1, 0, 0, 1, 0, 0, 0], dtype=np.int8)
    raise ValueError(rate_label)

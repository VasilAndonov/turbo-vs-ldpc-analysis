"""
Configuration file for the optimized LDPC simulation.

This project is written so that the LDPC results can be compared directly with
the turbo-code project:
- same information block length
- same nominal code rate
- same Eb/N0 sweep for the LDPC part
- same decoder iteration counts
"""

import numpy as np

RANDOM_SEED = 12
INFORMATION_BIT_COUNT = 256
CODE_RATE = 1.0 / 3.0
CODEWORD_BIT_COUNT = int(round(INFORMATION_BIT_COUNT / CODE_RATE))
PARITY_BIT_COUNT = CODEWORD_BIT_COUNT - INFORMATION_BIT_COUNT

INFORMATION_COLUMN_WEIGHT = 6
ROW_WEIGHT_PENALTY = 0.35

DECODER_ITERATION_LIST = [1, 2, 3, 4, 5, 6]
NORMALIZATION_FACTOR = 0.80

DEBUG_MODE = True
LDPC_EB_NO_DB = np.arange(-1.0, 1.76, 0.25, dtype=float)
MINIMUM_FRAME_COUNT = 80 if DEBUG_MODE else 200
MAXIMUM_FRAME_COUNT = 600 if DEBUG_MODE else 1500
TARGET_ERROR_COUNT = 160 if DEBUG_MODE else 500

SAVE_FIGURES = False
SAVE_PREFIX = "ldpc_optimized"

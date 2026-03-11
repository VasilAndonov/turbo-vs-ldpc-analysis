import numpy as np

# ============================================================
# Experiment settings
# ============================================================
# Keep these values aligned with the turbo-code project so that
# the BER curves can be compared on the same axes.
DEBUG_MODE = True
RANDOM_SEED = 12

# ============================================================
# Code dimensions
# ============================================================
# We keep the same number of information bits as in the turbo code.
# For a rate-1/3 code, the codeword length is three times larger.
INFORMATION_BIT_COUNT = 256
CODE_RATE = 1.0 / 3.0
CODEWORD_BIT_COUNT = int(round(INFORMATION_BIT_COUNT / CODE_RATE))
PARITY_BIT_COUNT = CODEWORD_BIT_COUNT - INFORMATION_BIT_COUNT

if PARITY_BIT_COUNT <= 0:
    raise ValueError(
        f"Parity bit count must be positive, got {PARITY_BIT_COUNT}. "
        "Check INFORMATION_BIT_COUNT and CODE_RATE."
    )

# ============================================================
# Sparse matrix construction
# ============================================================
# The parity-check matrix has the form H = [A | T].
# A is sparse and random. T is sparse lower-bidiagonal so that
# encoding is easy through forward substitution over GF(2).
INFORMATION_COLUMN_WEIGHT = 3

if INFORMATION_COLUMN_WEIGHT > PARITY_BIT_COUNT:
    raise ValueError(
        "INFORMATION_COLUMN_WEIGHT must be no larger than PARITY_BIT_COUNT."
    )

# ============================================================
# Iterative decoding settings
# ============================================================
ITERATION_LIST = [1, 2, 3, 4, 5, 6]
NORMALIZATION_FACTOR = 0.80  # normalized min-sum scaling

# ============================================================
# Simulation sweep
# ============================================================
EBN0_DECIBELS = np.arange(-1.0, 1.76, 0.25, dtype=float)

MINIMUM_FRAME_COUNT = 80 if DEBUG_MODE else 160
MAXIMUM_FRAME_COUNT = 500 if DEBUG_MODE else 1500
TARGET_ERROR_COUNT = 150 if DEBUG_MODE else 500

# ============================================================
# Plot/export settings
# ============================================================
SAVE_FIGURES = False
SAVE_PREFIX = "ldpc_comparison"

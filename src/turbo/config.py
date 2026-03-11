"""Configuration for the turbo-code simulation project.

This file collects all experiment settings in one place so the other modules stay focused on
algorithm logic.
"""

import numpy as np

# --------------------------------------------------------------------------------------
# Trellis and block configuration
# --------------------------------------------------------------------------------------
# The constituent recursive systematic convolutional encoder has memory 2, so the trellis
# has 2^2 = 4 states. We also use two tail bits to drive the encoder back to state zero.
ENCODER_MEMORY = 2
NUMBER_OF_TAIL_BITS = 2
NUMBER_OF_STATES = 4
INFORMATION_BLOCK_LENGTH = 256

# --------------------------------------------------------------------------------------
# Signal-to-noise ratio grids for the BER curves
# --------------------------------------------------------------------------------------
CONVOLUTIONAL_EB_NO_DB = np.arange(-4.0, 5.0, 1.0, dtype=float)
TURBO_EB_NO_DB = np.arange(-1.0, 1.76, 0.25, dtype=float)

# --------------------------------------------------------------------------------------
# Monte Carlo stopping rules
# --------------------------------------------------------------------------------------
# The lower bounds make sure each BER point uses at least some minimum averaging.
# The target-error stopping rule prevents very noisy BER estimates.
CONVOLUTIONAL_MINIMUM_FRAMES = 80
CONVOLUTIONAL_MAXIMUM_FRAMES = 400
CONVOLUTIONAL_TARGET_ERRORS = 400

TURBO_MINIMUM_FRAMES = 80
TURBO_MAXIMUM_FRAMES = 500
TURBO_TARGET_ERRORS = 150

# --------------------------------------------------------------------------------------
# Decoder settings
# --------------------------------------------------------------------------------------
DECODER_ITERATION_LIST = [1, 2, 3, 4, 5, 6]
EXTRINSIC_SCALING_FACTOR = 0.75

# --------------------------------------------------------------------------------------
# Plot settings
# --------------------------------------------------------------------------------------
SAVE_FIGURES = False
FIGURE_FILE_PREFIX = "turbo_literature"

# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------
RANDOM_SEED = 12

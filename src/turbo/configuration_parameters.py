import numpy as np

# ============================================================
# GENERAL CONFIGURATION PARAMETERS
# ============================================================

# Convolutional encoder memory order (number of previous bits used in feedback)
CONVOLUTIONAL_MEMORY_ORDER = 2  

# Number of tail bits to terminate the trellis to zero state
TAIL_BIT_COUNT = 2  

# Number of states in the convolutional trellis (2^memory)
NUMBER_OF_STATES = 4  

# Length of information blocks to be encoded
INFORMATION_BLOCK_LENGTH = 1024  

# Eb/N0 values for simulation (in dB) for convolutional code
CONVOLUTIONAL_EBN0_VALUES_DB = np.arange(-4, 5, 1, dtype=float)

# Eb/N0 values for simulation (in dB) for turbo code
TURBO_EBN0_VALUES_DB = np.arange(-1.0, 1.76, 0.25, dtype=float)

# Simulation limits for convolutional code
CONVOLUTIONAL_MINIMUM_FRAMES = 80  # Minimum frames per Eb/N0
CONVOLUTIONAL_MAXIMUM_FRAMES = 400  # Maximum frames per Eb/N0
CONVOLUTIONAL_TARGET_ERRORS = 400  # Stop after this many bit errors

# Simulation limits for turbo code
TURBO_MINIMUM_FRAMES = 80
TURBO_MAXIMUM_FRAMES = 500
TURBO_TARGET_ERRORS = 150

# Number of turbo decoder iterations
TURBO_DECODER_ITERATION_COUNTS = [1,2,3,4,5,6]

# Random seed for reproducibility
RANDOM_SEED = 12

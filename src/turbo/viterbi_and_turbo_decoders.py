import numpy as np
from configuration_parameters import *
from convolutional_and_turbo_encoders import *

# ============================================================
# VITERBI DECODER FUNCTION
# ============================================================

def viterbi_decode_convolutional_code(received_signal, number_of_information_bits):
    """
    Decode a convolutionally encoded sequence using the Viterbi algorithm.
    
    Args:
        received_signal: received BPSK symbols (float array)
        number_of_information_bits: number of original info bits to recover
    
    Returns:
        decoded_bits: estimated information bits
    """

    number_of_steps = len(received_signal) // 2  # two output bits per input bit

    # Initialize path metrics (high initial cost)
    path_metric = np.full((number_of_steps+1, NUMBER_OF_STATES), 1e30)

    # Tables to store the best previous state and input
    previous_state = np.full((number_of_steps+1, NUMBER_OF_STATES), -1)
    previous_input = np.full((number_of_steps+1, NUMBER_OF_STATES), -1)

    # Start from state 0 with zero metric
    path_metric[0, 0] = 0.0

    # Forward recursion
    for step in range(number_of_steps):

        received0 = received_signal[2*step]
        received1 = received_signal[2*step+1]

        for state in range(NUMBER_OF_STATES):

            metric = path_metric[step, state]

            if metric > 1e20:
                continue  # skip unreachable states

            # Current register bits
            register0 = (state >> 1) & 1
            register1 = state & 1

            for input_bit in (0, 1):

                # Compute outputs
                coded0 = input_bit ^ register0 ^ register1
                coded1 = input_bit ^ register1

                # Compute next state
                next_state = (input_bit << 1) | register0

                # Expected BPSK symbols
                expected0 = 1.0 if coded0 == 0 else -1.0
                expected1 = 1.0 if coded1 == 0 else -1.0

                # Branch metric (Euclidean distance)
                branch_metric = (received0 - expected0)**2 + (received1 - expected1)**2

                candidate_metric = metric + branch_metric

                # Update if this path is better
                if candidate_metric < path_metric[step+1, next_state]:
                    path_metric[step+1, next_state] = candidate_metric
                    previous_state[step+1, next_state] = state
                    previous_input[step+1, next_state] = input_bit

    # Traceback
    state = 0
    decoded_bits = []

    for step in range(number_of_steps, 0, -1):
        decoded_bits.append(previous_input[step, state])
        state = previous_state[step, state]

    decoded_bits.reverse()

    return np.array(decoded_bits[:number_of_information_bits], dtype=np.int8)

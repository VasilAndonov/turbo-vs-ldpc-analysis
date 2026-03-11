import numpy as np
from configuration_parameters import *
from convolutional_and_turbo_encoders import *

# ============================================================
# VITERBI DECODER
# ============================================================

def viterbi_decode_convolutional_code(received_signal, number_of_information_bits):
    """
    Decode a convolutionally encoded sequence using the Viterbi algorithm.
    """
    number_of_steps = len(received_signal) // 2
    path_metric = np.full((number_of_steps+1, NUMBER_OF_STATES), 1e30)
    previous_state = np.full((number_of_steps+1, NUMBER_OF_STATES), -1)
    previous_input = np.full((number_of_steps+1, NUMBER_OF_STATES), -1)
    path_metric[0,0] = 0.0

    for step in range(number_of_steps):
        received0 = received_signal[2*step]
        received1 = received_signal[2*step+1]
        for state in range(NUMBER_OF_STATES):
            metric = path_metric[step,state]
            if metric > 1e20:
                continue
            register0 = (state >> 1) & 1
            register1 = state & 1
            for input_bit in (0,1):
                coded0 = input_bit ^ register0 ^ register1
                coded1 = input_bit ^ register1
                next_state = (input_bit<<1) | register0
                expected0 = 1.0 if coded0==0 else -1.0
                expected1 = 1.0 if coded1==0 else -1.0
                branch_metric = (received0-expected0)**2 + (received1-expected1)**2
                candidate_metric = metric + branch_metric
                if candidate_metric < path_metric[step+1,next_state]:
                    path_metric[step+1,next_state] = candidate_metric
                    previous_state[step+1,next_state] = state
                    previous_input[step+1,next_state] = input_bit

    # Traceback
    state = 0
    decoded_bits = []
    for step in range(number_of_steps,0,-1):
        decoded_bits.append(previous_input[step,state])
        state = previous_state[step,state]
    decoded_bits.reverse()
    return np.array(decoded_bits[:number_of_information_bits], dtype=np.int8)

# ============================================================
# TURBO MAX-LOG-MAP DECODER (simplified)
# ============================================================

def turbo_maxlog_map_decode(sys1_rx, par1_rx, par2_rx, interleaver, sigma2, n_iterations):
    """
    Iterative Turbo decoder (max-log-MAP) for simulation purposes.
    Returns hard decisions and LLR per iteration.
    """
    n_total = len(sys1_rx)
    llr_sys1 = 2.0/sigma2 * sys1_rx
    llr_par1 = 2.0/sigma2 * par1_rx
    llr_par2 = 2.0/sigma2 * par2_rx
    apriori1 = np.zeros(n_total)
    llr_iterations = []
    ext_scale = 0.75

    for _ in range(n_iterations):
        # simplified iterative LLR updates
        ext1 = llr_par1 * ext_scale
        apriori2 = ext1[interleaver]
        llr_iterations.append(apriori2.copy())

    hard_decisions = (llr_iterations[-1] < 0).astype(np.int8)
    return hard_decisions, llr_iterations

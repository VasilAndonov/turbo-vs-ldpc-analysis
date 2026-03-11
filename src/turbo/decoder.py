"""Decoder-side functions for the turbo-code simulation.

This file contains:
1. A soft-decision Viterbi decoder for the rate-1/2 convolutional baseline.
2. A terminated max-log-MAP constituent decoder.
3. An iterative turbo decoder that exchanges extrinsic information between two constituent
   decoders.
"""

import numpy as np

from config import (
    ENCODER_MEMORY,
    EXTRINSIC_SCALING_FACTOR,
    INFORMATION_BLOCK_LENGTH,
    NUMBER_OF_STATES,
)
from encoder import (
    INFORMATION_BIT_SIGN_TABLE,
    NEXT_STATE_TABLE,
    PARITY_BIT_SIGN_TABLE,
    PREDECESSOR_STATE_TABLE,
    SUCCESSOR_STATE_TABLE,
)


# --------------------------------------------------------------------------------------
# Soft-decision Viterbi decoder for the (7,5) convolutional code
# --------------------------------------------------------------------------------------
# The decoder works with noisy BPSK symbols. Each branch metric is the Euclidean distance
# between the received pair and the expected coded-symbol pair.

def decode_convolutional_75_viterbi(received_symbols, number_of_information_bits):
    received_symbols = np.asarray(received_symbols, dtype=float)
    number_of_trellis_steps = len(received_symbols) // 2

    path_metric_table = np.full((number_of_trellis_steps + 1, NUMBER_OF_STATES), 1e30)
    predecessor_state_choice = np.full((number_of_trellis_steps + 1, NUMBER_OF_STATES), -1, dtype=int)
    predecessor_input_choice = np.full((number_of_trellis_steps + 1, NUMBER_OF_STATES), -1, dtype=int)
    path_metric_table[0, 0] = 0.0

    for step_index in range(number_of_trellis_steps):
        received_symbol_one = received_symbols[2 * step_index]
        received_symbol_two = received_symbols[2 * step_index + 1]

        for state_index in range(NUMBER_OF_STATES):
            current_metric = path_metric_table[step_index, state_index]
            if current_metric > 1e20:
                continue

            oldest_memory_bit = (state_index >> 1) & 1
            newest_memory_bit = state_index & 1

            for input_bit in (0, 1):
                first_coded_bit = input_bit ^ oldest_memory_bit ^ newest_memory_bit
                second_coded_bit = input_bit ^ newest_memory_bit
                next_state_index = (input_bit << 1) | oldest_memory_bit

                expected_symbol_one = 1.0 if first_coded_bit == 0 else -1.0
                expected_symbol_two = 1.0 if second_coded_bit == 0 else -1.0

                branch_metric = (
                    (received_symbol_one - expected_symbol_one) ** 2
                    + (received_symbol_two - expected_symbol_two) ** 2
                )
                candidate_metric = current_metric + branch_metric

                if candidate_metric < path_metric_table[step_index + 1, next_state_index]:
                    path_metric_table[step_index + 1, next_state_index] = candidate_metric
                    predecessor_state_choice[step_index + 1, next_state_index] = state_index
                    predecessor_input_choice[step_index + 1, next_state_index] = input_bit

    final_state_index = 0
    decoded_bit_list = []
    for step_index in range(number_of_trellis_steps, 0, -1):
        decoded_bit_list.append(predecessor_input_choice[step_index, final_state_index])
        final_state_index = predecessor_state_choice[step_index, final_state_index]

    decoded_bit_list.reverse()
    return np.array(decoded_bit_list[:number_of_information_bits], dtype=np.int8)


# --------------------------------------------------------------------------------------
# Terminated max-log-MAP constituent decoder
# --------------------------------------------------------------------------------------
# This is the soft-input soft-output block used inside turbo decoding. It computes the
# a-posteriori information and then removes the systematic and a-priori parts to obtain the
# extrinsic information passed to the other decoder.

def decode_constituent_max_log_map_terminated(
    systematic_llr_values,
    parity_llr_values,
    apriori_llr_values,
):
    systematic_llr_values = np.asarray(systematic_llr_values, dtype=float)
    parity_llr_values = np.asarray(parity_llr_values, dtype=float)
    apriori_llr_values = np.asarray(apriori_llr_values, dtype=float)

    number_of_time_steps = len(systematic_llr_values)
    negative_infinity = -1e15

    forward_metric_table = np.full((number_of_time_steps + 1, NUMBER_OF_STATES), negative_infinity, dtype=float)
    backward_metric_table = np.full((number_of_time_steps + 1, NUMBER_OF_STATES), negative_infinity, dtype=float)
    forward_metric_table[0, 0] = 0.0
    backward_metric_table[number_of_time_steps, 0] = 0.0

    zero_branch_metric_table = np.zeros((number_of_time_steps, NUMBER_OF_STATES), dtype=float)
    one_branch_metric_table = np.zeros((number_of_time_steps, NUMBER_OF_STATES), dtype=float)

    for time_index in range(number_of_time_steps):
        for state_index in range(NUMBER_OF_STATES):
            zero_branch_metric_table[time_index, state_index] = 0.5 * (
                (systematic_llr_values[time_index] + apriori_llr_values[time_index])
                * INFORMATION_BIT_SIGN_TABLE[state_index, 0]
                + parity_llr_values[time_index] * PARITY_BIT_SIGN_TABLE[state_index, 0]
            )
            one_branch_metric_table[time_index, state_index] = 0.5 * (
                (systematic_llr_values[time_index] + apriori_llr_values[time_index])
                * INFORMATION_BIT_SIGN_TABLE[state_index, 1]
                + parity_llr_values[time_index] * PARITY_BIT_SIGN_TABLE[state_index, 1]
            )

    for time_index in range(number_of_time_steps):
        for next_state_index in range(NUMBER_OF_STATES):
            best_metric = negative_infinity
            for predecessor_state_index, input_bit in PREDECESSOR_STATE_TABLE[next_state_index]:
                branch_metric = (
                    zero_branch_metric_table[time_index, predecessor_state_index]
                    if input_bit == 0
                    else one_branch_metric_table[time_index, predecessor_state_index]
                )
                candidate_metric = forward_metric_table[time_index, predecessor_state_index] + branch_metric
                if candidate_metric > best_metric:
                    best_metric = candidate_metric
            forward_metric_table[time_index + 1, next_state_index] = best_metric

    for time_index in range(number_of_time_steps - 1, -1, -1):
        for state_index in range(NUMBER_OF_STATES):
            best_metric = negative_infinity
            for next_state_index, input_bit in SUCCESSOR_STATE_TABLE[state_index]:
                branch_metric = (
                    zero_branch_metric_table[time_index, state_index]
                    if input_bit == 0
                    else one_branch_metric_table[time_index, state_index]
                )
                candidate_metric = backward_metric_table[time_index + 1, next_state_index] + branch_metric
                if candidate_metric > best_metric:
                    best_metric = candidate_metric
            backward_metric_table[time_index, state_index] = best_metric

    posterior_llr_values = np.zeros(number_of_time_steps, dtype=float)
    for time_index in range(number_of_time_steps):
        best_zero_metric = negative_infinity
        best_one_metric = negative_infinity

        for state_index in range(NUMBER_OF_STATES):
            next_state_for_zero = NEXT_STATE_TABLE[state_index, 0]
            next_state_for_one = NEXT_STATE_TABLE[state_index, 1]

            zero_path_metric = (
                forward_metric_table[time_index, state_index]
                + zero_branch_metric_table[time_index, state_index]
                + backward_metric_table[time_index + 1, next_state_for_zero]
            )
            one_path_metric = (
                forward_metric_table[time_index, state_index]
                + one_branch_metric_table[time_index, state_index]
                + backward_metric_table[time_index + 1, next_state_for_one]
            )

            if zero_path_metric > best_zero_metric:
                best_zero_metric = zero_path_metric
            if one_path_metric > best_one_metric:
                best_one_metric = one_path_metric

        posterior_llr_values[time_index] = best_zero_metric - best_one_metric

    extrinsic_llr_values = posterior_llr_values - systematic_llr_values - apriori_llr_values
    return posterior_llr_values, extrinsic_llr_values


# --------------------------------------------------------------------------------------
# Iterative turbo decoder
# --------------------------------------------------------------------------------------
# Decoder one works in the original bit order. Decoder two works in the interleaved order.
# After each half-iteration, only extrinsic information is exchanged.

def turbo_decode(
    received_systematic_stream_one,
    received_parity_stream_one,
    received_parity_stream_two,
    interleaver_pattern,
    noise_variance,
    number_of_iterations,
):
    total_stream_length = len(received_systematic_stream_one)
    llr_scale = 2.0 / noise_variance

    systematic_llr_stream_one = llr_scale * received_systematic_stream_one
    parity_llr_stream_one = llr_scale * received_parity_stream_one
    parity_llr_stream_two = llr_scale * received_parity_stream_two

    apriori_llr_stream_one = np.zeros(total_stream_length, dtype=float)
    iteration_llr_history = []

    for _ in range(number_of_iterations):
        _, extrinsic_llr_stream_one = decode_constituent_max_log_map_terminated(
            systematic_llr_stream_one,
            parity_llr_stream_one,
            apriori_llr_stream_one,
        )

        apriori_llr_stream_two = np.zeros(total_stream_length, dtype=float)
        apriori_llr_stream_two[:INFORMATION_BLOCK_LENGTH] = (
            EXTRINSIC_SCALING_FACTOR * extrinsic_llr_stream_one[interleaver_pattern]
        )

        interleaved_systematic_llr_stream = np.zeros(total_stream_length, dtype=float)
        interleaved_systematic_llr_stream[:INFORMATION_BLOCK_LENGTH] = systematic_llr_stream_one[interleaver_pattern]

        posterior_llr_stream_two_interleaved, extrinsic_llr_stream_two_interleaved = (
            decode_constituent_max_log_map_terminated(
                interleaved_systematic_llr_stream,
                parity_llr_stream_two,
                apriori_llr_stream_two,
            )
        )

        posterior_information_llr = np.zeros(INFORMATION_BLOCK_LENGTH, dtype=float)
        posterior_information_llr[interleaver_pattern] = posterior_llr_stream_two_interleaved[:INFORMATION_BLOCK_LENGTH]

        extrinsic_information_llr = np.zeros(INFORMATION_BLOCK_LENGTH, dtype=float)
        extrinsic_information_llr[interleaver_pattern] = extrinsic_llr_stream_two_interleaved[:INFORMATION_BLOCK_LENGTH]

        iteration_llr_history.append(posterior_information_llr.copy())
        apriori_llr_stream_one[:INFORMATION_BLOCK_LENGTH] = EXTRINSIC_SCALING_FACTOR * extrinsic_information_llr

    final_hard_decision_bits = (iteration_llr_history[-1] < 0.0).astype(np.int8)
    return final_hard_decision_bits, iteration_llr_history

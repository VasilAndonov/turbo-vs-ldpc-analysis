"""
Turbo and convolutional decoders.
"""

import numpy as np

from config import INFORMATION_BLOCK_LENGTH, RSC_STATE_COUNT
from encoder import (
    INPUT_SIGN,
    INTERLEAVER,
    NEXT_STATE,
    PARITY_SIGN,
    PREDECESSOR_LIST,
    SUCCESSOR_LIST,
)

def decode_viterbi_75(received_values, information_length):
    received_values = np.asarray(received_values, dtype=float)
    step_count = len(received_values) // 2

    path_metric = np.full((step_count + 1, RSC_STATE_COUNT), 1e30, dtype=float)
    predecessor_state = np.full((step_count + 1, RSC_STATE_COUNT), -1, dtype=int)
    predecessor_input = np.full((step_count + 1, RSC_STATE_COUNT), -1, dtype=int)

    path_metric[0, 0] = 0.0

    for step_index in range(step_count):
        received_0 = received_values[2 * step_index]
        received_1 = received_values[2 * step_index + 1]

        for state in range(RSC_STATE_COUNT):
            current_metric = path_metric[step_index, state]
            if current_metric > 1e20:
                continue

            register_bit_0 = (state >> 1) & 1
            register_bit_1 = state & 1

            for information_bit in (0, 1):
                output_0 = information_bit ^ register_bit_0 ^ register_bit_1
                output_1 = information_bit ^ register_bit_1
                next_state = (information_bit << 1) | register_bit_0

                expected_0 = 1.0 if output_0 == 0 else -1.0
                expected_1 = 1.0 if output_1 == 0 else -1.0

                branch_metric = (received_0 - expected_0) ** 2 + (received_1 - expected_1) ** 2
                candidate_metric = current_metric + branch_metric

                if candidate_metric < path_metric[step_index + 1, next_state]:
                    path_metric[step_index + 1, next_state] = candidate_metric
                    predecessor_state[step_index + 1, next_state] = state
                    predecessor_input[step_index + 1, next_state] = information_bit

    decoded_bits = []
    state = 0
    for step_index in range(step_count, 0, -1):
        decoded_bits.append(predecessor_input[step_index, state])
        state = predecessor_state[step_index, state]

    decoded_bits.reverse()
    return np.array(decoded_bits[:information_length], dtype=np.int8)


def maxlogmap_decode_terminated(systematic_llr, parity_llr, apriori_llr):
    systematic_llr = np.asarray(systematic_llr, dtype=float)
    parity_llr = np.asarray(parity_llr, dtype=float)
    apriori_llr = np.asarray(apriori_llr, dtype=float)

    symbol_count = len(systematic_llr)
    negative_infinity = -1e15

    alpha = np.full((symbol_count + 1, RSC_STATE_COUNT), negative_infinity, dtype=float)
    beta = np.full((symbol_count + 1, RSC_STATE_COUNT), negative_infinity, dtype=float)
    alpha[0, 0] = 0.0
    beta[symbol_count, 0] = 0.0

    gamma_for_zero = np.zeros((symbol_count, RSC_STATE_COUNT), dtype=float)
    gamma_for_one = np.zeros((symbol_count, RSC_STATE_COUNT), dtype=float)

    for symbol_index in range(symbol_count):
        for state in range(RSC_STATE_COUNT):
            gamma_for_zero[symbol_index, state] = 0.5 * (
                (systematic_llr[symbol_index] + apriori_llr[symbol_index]) * INPUT_SIGN[state, 0]
                + parity_llr[symbol_index] * PARITY_SIGN[state, 0]
            )
            gamma_for_one[symbol_index, state] = 0.5 * (
                (systematic_llr[symbol_index] + apriori_llr[symbol_index]) * INPUT_SIGN[state, 1]
                + parity_llr[symbol_index] * PARITY_SIGN[state, 1]
            )

    for symbol_index in range(symbol_count):
        for next_state in range(RSC_STATE_COUNT):
            best_value = negative_infinity
            for state, input_bit in PREDECESSOR_LIST[next_state]:
                gamma = gamma_for_zero[symbol_index, state] if input_bit == 0 else gamma_for_one[symbol_index, state]
                candidate = alpha[symbol_index, state] + gamma
                if candidate > best_value:
                    best_value = candidate
            alpha[symbol_index + 1, next_state] = best_value

    for symbol_index in range(symbol_count - 1, -1, -1):
        for state in range(RSC_STATE_COUNT):
            best_value = negative_infinity
            for next_state, input_bit in SUCCESSOR_LIST[state]:
                gamma = gamma_for_zero[symbol_index, state] if input_bit == 0 else gamma_for_one[symbol_index, state]
                candidate = beta[symbol_index + 1, next_state] + gamma
                if candidate > best_value:
                    best_value = candidate
            beta[symbol_index, state] = best_value

    posterior_llr = np.zeros(symbol_count, dtype=float)
    for symbol_index in range(symbol_count):
        best_zero = negative_infinity
        best_one = negative_infinity

        for state in range(RSC_STATE_COUNT):
            next_state_zero = NEXT_STATE[state, 0]
            next_state_one = NEXT_STATE[state, 1]

            candidate_zero = alpha[symbol_index, state] + gamma_for_zero[symbol_index, state] + beta[symbol_index + 1, next_state_zero]
            candidate_one = alpha[symbol_index, state] + gamma_for_one[symbol_index, state] + beta[symbol_index + 1, next_state_one]

            if candidate_zero > best_zero:
                best_zero = candidate_zero
            if candidate_one > best_one:
                best_one = candidate_one

        posterior_llr[symbol_index] = best_zero - best_one

    extrinsic_llr = posterior_llr - systematic_llr - apriori_llr
    return posterior_llr, extrinsic_llr


def decode_turbo(
    received_systematic_stream_1,
    received_parity_stream_1_full,
    received_parity_stream_2_full,
    noise_variance,
    iteration_count,
):
    total_length = len(received_systematic_stream_1)
    channel_reliability = 2.0 / noise_variance

    systematic_llr_1 = channel_reliability * np.asarray(received_systematic_stream_1, dtype=float)
    parity_llr_1 = channel_reliability * np.asarray(received_parity_stream_1_full, dtype=float)
    parity_llr_2 = channel_reliability * np.asarray(received_parity_stream_2_full, dtype=float)

    apriori_llr_decoder_1 = np.zeros(total_length, dtype=float)
    llr_history = []
    extrinsic_scale = 0.75

    for _ in range(iteration_count):
        _, extrinsic_llr_1 = maxlogmap_decode_terminated(
            systematic_llr_1,
            parity_llr_1,
            apriori_llr_decoder_1,
        )

        apriori_llr_decoder_2 = np.zeros(total_length, dtype=float)
        apriori_llr_decoder_2[:INFORMATION_BLOCK_LENGTH] = extrinsic_scale * extrinsic_llr_1[INTERLEAVER]

        interleaved_systematic_llr = np.zeros(total_length, dtype=float)
        interleaved_systematic_llr[:INFORMATION_BLOCK_LENGTH] = systematic_llr_1[INTERLEAVER]

        posterior_llr_2_interleaved, extrinsic_llr_2_interleaved = maxlogmap_decode_terminated(
            interleaved_systematic_llr,
            parity_llr_2,
            apriori_llr_decoder_2,
        )

        posterior_information_llr = np.zeros(INFORMATION_BLOCK_LENGTH, dtype=float)
        posterior_information_llr[INTERLEAVER] = posterior_llr_2_interleaved[:INFORMATION_BLOCK_LENGTH]

        extrinsic_information_llr = np.zeros(INFORMATION_BLOCK_LENGTH, dtype=float)
        extrinsic_information_llr[INTERLEAVER] = extrinsic_llr_2_interleaved[:INFORMATION_BLOCK_LENGTH]

        llr_history.append(posterior_information_llr.copy())
        apriori_llr_decoder_1[:INFORMATION_BLOCK_LENGTH] = extrinsic_scale * extrinsic_information_llr

    hard_information_bits = (llr_history[-1] < 0.0).astype(np.int8)
    return hard_information_bits, llr_history

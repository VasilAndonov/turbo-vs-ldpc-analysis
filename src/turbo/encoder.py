"""
Turbo encoder utilities.

This file provides:
1) the recursive systematic convolutional constituent trellis,
2) the terminated constituent encoder,
3) puncturing helpers for multi-rate turbo transmission,
4) a conventional rate-1/2 convolutional encoder for the baseline curve.
"""

import numpy as np

from config import (
    INFORMATION_BLOCK_LENGTH,
    RANDOM_SEED,
    RSC_MEMORY,
    RSC_STATE_COUNT,
    RSC_TAIL_LENGTH,
    get_puncture_definition,
)


def build_rsc_tables():
    """
    Build state transition and parity tables for a 4-state RSC code.

    Feedback polynomial: 101
    Feedforward polynomial: 111
    """
    next_state = np.zeros((RSC_STATE_COUNT, 2), dtype=int)
    parity_bit = np.zeros((RSC_STATE_COUNT, 2), dtype=int)
    input_sign = np.zeros((RSC_STATE_COUNT, 2), dtype=float)
    parity_sign = np.zeros((RSC_STATE_COUNT, 2), dtype=float)
    predecessor_list = [[] for _ in range(RSC_STATE_COUNT)]
    successor_list = [[] for _ in range(RSC_STATE_COUNT)]

    for state in range(RSC_STATE_COUNT):
        register_bit_0 = (state >> 1) & 1
        register_bit_1 = state & 1

        for information_bit in (0, 1):
            recursive_bit = information_bit ^ register_bit_1
            parity = recursive_bit ^ register_bit_0 ^ register_bit_1

            next_register_bit_0 = recursive_bit
            next_register_bit_1 = register_bit_0
            next_state_value = (next_register_bit_0 << 1) | next_register_bit_1

            next_state[state, information_bit] = next_state_value
            parity_bit[state, information_bit] = parity
            input_sign[state, information_bit] = 1.0 if information_bit == 0 else -1.0
            parity_sign[state, information_bit] = 1.0 if parity == 0 else -1.0

            predecessor_list[next_state_value].append((state, information_bit))
            successor_list[state].append((next_state_value, information_bit))

    return next_state, parity_bit, input_sign, parity_sign, predecessor_list, successor_list


(
    NEXT_STATE,
    PARITY_BIT,
    INPUT_SIGN,
    PARITY_SIGN,
    PREDECESSOR_LIST,
    SUCCESSOR_LIST,
) = build_rsc_tables()


def build_interleaver():
    random_generator = np.random.default_rng(RANDOM_SEED)
    permutation = random_generator.permutation(INFORMATION_BLOCK_LENGTH)
    inverse_permutation = np.argsort(permutation)
    return permutation.astype(np.int64), inverse_permutation.astype(np.int64)


INTERLEAVER, DEINTERLEAVER = build_interleaver()


def tail_bits_for_zero_termination(current_state):
    """
    Compute the input bits that drive the RSC encoder back to the zero state.
    """
    tail_sequence = []
    state = int(current_state)

    for _ in range(RSC_TAIL_LENGTH):
        register_bit_1 = state & 1
        information_bit = register_bit_1
        tail_sequence.append(information_bit)
        state = NEXT_STATE[state, information_bit]

    return tail_sequence


def encode_rsc_terminated(information_bits):
    """
    Encode one input block with terminated RSC encoding.

    Returns:
    - systematic_bits
    - parity_bits
    """
    information_bits = np.asarray(information_bits, dtype=np.int8)

    systematic_bits = []
    parity_bits = []
    state = 0

    for bit_value in information_bits:
        bit_value = int(bit_value)
        systematic_bits.append(bit_value)
        parity_bits.append(PARITY_BIT[state, bit_value])
        state = NEXT_STATE[state, bit_value]

    tail_sequence = tail_bits_for_zero_termination(state)
    for bit_value in tail_sequence:
        systematic_bits.append(int(bit_value))
        parity_bits.append(PARITY_BIT[state, bit_value])
        state = NEXT_STATE[state, bit_value]

    return np.array(systematic_bits, dtype=np.int8), np.array(parity_bits, dtype=np.int8)


def turbo_encode_full_streams(information_bits):
    """
    Generate the full systematic and parity streams before puncturing.
    """
    information_bits = np.asarray(information_bits, dtype=np.int8)

    systematic_stream_1, parity_stream_1 = encode_rsc_terminated(information_bits)
    interleaved_information_bits = information_bits[INTERLEAVER]
    systematic_stream_2, parity_stream_2 = encode_rsc_terminated(interleaved_information_bits)

    return systematic_stream_1, parity_stream_1, systematic_stream_2, parity_stream_2


def build_puncture_mask(total_length):
    """
    Build repeated keep masks for the two parity streams.
    """
    puncture_definition = get_puncture_definition()
    pattern_1 = puncture_definition["parity_1_pattern"]
    pattern_2 = puncture_definition["parity_2_pattern"]

    parity_keep_mask_1 = np.resize(pattern_1, total_length).astype(np.int8)
    parity_keep_mask_2 = np.resize(pattern_2, total_length).astype(np.int8)
    return parity_keep_mask_1, parity_keep_mask_2


def turbo_encode_transmitted_symbols(information_bits):
    """
    Return the transmitted turbo streams after puncturing.

    The decoder can reconstruct punctured positions by inserting zero LLR.
    """
    systematic_stream_1, parity_stream_1, systematic_stream_2, parity_stream_2 = turbo_encode_full_streams(information_bits)
    total_length = len(systematic_stream_1)
    parity_keep_mask_1, parity_keep_mask_2 = build_puncture_mask(total_length)

    transmitted_parity_stream_1 = parity_stream_1[parity_keep_mask_1 == 1]
    transmitted_parity_stream_2 = parity_stream_2[parity_keep_mask_2 == 1]

    return {
        "systematic_stream_1": systematic_stream_1,
        "parity_stream_1_full": parity_stream_1,
        "parity_stream_2_full": parity_stream_2,
        "transmitted_parity_stream_1": transmitted_parity_stream_1,
        "transmitted_parity_stream_2": transmitted_parity_stream_2,
        "parity_keep_mask_1": parity_keep_mask_1,
        "parity_keep_mask_2": parity_keep_mask_2,
    }


def depuncture_received_parity(transmitted_received_values, keep_mask):
    """
    Reinsert punctured positions as zero-valued channel samples.
    """
    depunctured_values = np.zeros(len(keep_mask), dtype=float)
    depunctured_values[keep_mask == 1] = np.asarray(transmitted_received_values, dtype=float)
    return depunctured_values


def encode_convolutional_75(information_bits):
    """
    Conventional non-recursive rate-1/2 convolutional encoder with generators (7,5).
    """
    information_bits = np.asarray(information_bits, dtype=np.int8)
    information_bits = np.concatenate([information_bits, np.zeros(RSC_MEMORY, dtype=np.int8)])

    register_bit_0 = 0
    register_bit_1 = 0
    encoded_bits = []

    for input_bit in information_bits:
        input_bit = int(input_bit)
        output_0 = input_bit ^ register_bit_0 ^ register_bit_1
        output_1 = input_bit ^ register_bit_1
        encoded_bits.extend((output_0, output_1))

        register_bit_1 = register_bit_0
        register_bit_0 = input_bit

    return np.array(encoded_bits, dtype=np.int8)

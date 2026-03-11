"""Encoder-side functions for the turbo-code simulation.

This module contains two encoder families:
1. A terminated rate-1/2 convolutional encoder with generator pair (7,5) in octal.
2. A terminated recursive systematic convolutional encoder used inside the turbo code.

The turbo encoder is built from two identical recursive systematic encoders separated by a
random interleaver.
"""

import numpy as np

from config import ENCODER_MEMORY, NUMBER_OF_TAIL_BITS, NUMBER_OF_STATES


# --------------------------------------------------------------------------------------
# Recursive systematic convolutional trellis tables
# --------------------------------------------------------------------------------------
# These tables fully describe the constituent encoder used by the turbo code.
# Feedback polynomial: 101
# Feedforward polynomial: 111
# The signs are stored in BPSK form so the decoder can evaluate branch metrics directly.

def build_recursive_systematic_tables():
    next_state_table = np.zeros((NUMBER_OF_STATES, 2), dtype=int)
    parity_bit_table = np.zeros((NUMBER_OF_STATES, 2), dtype=int)
    information_bit_sign_table = np.zeros((NUMBER_OF_STATES, 2), dtype=float)
    parity_bit_sign_table = np.zeros((NUMBER_OF_STATES, 2), dtype=float)
    predecessor_state_table = [[] for _ in range(NUMBER_OF_STATES)]
    successor_state_table = [[] for _ in range(NUMBER_OF_STATES)]

    for state_index in range(NUMBER_OF_STATES):
        oldest_memory_bit = (state_index >> 1) & 1
        newest_memory_bit = state_index & 1

        for input_bit in (0, 1):
            recursive_input_bit = input_bit ^ newest_memory_bit
            parity_bit = recursive_input_bit ^ oldest_memory_bit ^ newest_memory_bit

            next_oldest_memory_bit = recursive_input_bit
            next_newest_memory_bit = oldest_memory_bit
            next_state_index = (next_oldest_memory_bit << 1) | next_newest_memory_bit

            next_state_table[state_index, input_bit] = next_state_index
            parity_bit_table[state_index, input_bit] = parity_bit
            information_bit_sign_table[state_index, input_bit] = 1.0 if input_bit == 0 else -1.0
            parity_bit_sign_table[state_index, input_bit] = 1.0 if parity_bit == 0 else -1.0

            predecessor_state_table[next_state_index].append((state_index, input_bit))
            successor_state_table[state_index].append((next_state_index, input_bit))

    return (
        next_state_table,
        parity_bit_table,
        information_bit_sign_table,
        parity_bit_sign_table,
        predecessor_state_table,
        successor_state_table,
    )


(
    NEXT_STATE_TABLE,
    PARITY_BIT_TABLE,
    INFORMATION_BIT_SIGN_TABLE,
    PARITY_BIT_SIGN_TABLE,
    PREDECESSOR_STATE_TABLE,
    SUCCESSOR_STATE_TABLE,
) = build_recursive_systematic_tables()


# --------------------------------------------------------------------------------------
# Tail-bit generation for the constituent encoder
# --------------------------------------------------------------------------------------
# After all information bits are processed, these extra bits drive the encoder back to the
# all-zero state. This makes the trellis termination explicit and helps the decoder.

def compute_tail_bits(current_state):
    tail_bit_list = []
    state_index = current_state

    for _ in range(NUMBER_OF_TAIL_BITS):
        newest_memory_bit = state_index & 1
        input_bit = newest_memory_bit
        tail_bit_list.append(input_bit)
        state_index = NEXT_STATE_TABLE[state_index, input_bit]

    return tail_bit_list


# --------------------------------------------------------------------------------------
# Terminated recursive systematic encoder
# --------------------------------------------------------------------------------------
# The output contains the systematic stream and one parity stream. Tail bits are appended
# to both streams so the final state is zero.

def encode_recursive_systematic_terminated(information_bits):
    information_bits = np.asarray(information_bits, dtype=np.int8)

    systematic_output_bits = []
    parity_output_bits = []
    state_index = 0

    for information_bit in information_bits:
        systematic_output_bits.append(int(information_bit))
        parity_output_bits.append(PARITY_BIT_TABLE[state_index, information_bit])
        state_index = NEXT_STATE_TABLE[state_index, information_bit]

    tail_bit_list = compute_tail_bits(state_index)
    for tail_bit in tail_bit_list:
        systematic_output_bits.append(int(tail_bit))
        parity_output_bits.append(PARITY_BIT_TABLE[state_index, tail_bit])
        state_index = NEXT_STATE_TABLE[state_index, tail_bit]

    return np.array(systematic_output_bits, dtype=np.int8), np.array(parity_output_bits, dtype=np.int8)


# --------------------------------------------------------------------------------------
# Turbo encoder
# --------------------------------------------------------------------------------------
# The turbo encoder uses two identical constituent encoders. The first sees the original
# information sequence and the second sees an interleaved version of the same sequence.

def turbo_encode(information_bits, interleaver_pattern):
    information_bits = np.asarray(information_bits, dtype=np.int8)

    first_systematic_stream, first_parity_stream = encode_recursive_systematic_terminated(information_bits)

    interleaved_information_bits = information_bits[interleaver_pattern]
    second_systematic_stream, second_parity_stream = encode_recursive_systematic_terminated(interleaved_information_bits)

    return (
        first_systematic_stream,
        first_parity_stream,
        second_systematic_stream,
        second_parity_stream,
    )


# --------------------------------------------------------------------------------------
# Baseline convolutional encoder with generators (7,5)
# --------------------------------------------------------------------------------------
# This encoder is used only for the comparison curve against the turbo code.
# It appends ENCODER_MEMORY zeros to terminate the trellis.

def encode_convolutional_75(information_bits):
    information_bits = np.asarray(information_bits, dtype=np.int8)
    terminated_information_bits = np.concatenate(
        [information_bits, np.zeros(ENCODER_MEMORY, dtype=np.int8)]
    )

    first_memory_bit = 0
    second_memory_bit = 0
    coded_bit_list = []

    for input_bit in terminated_information_bits:
        first_output_bit = input_bit ^ first_memory_bit ^ second_memory_bit
        second_output_bit = input_bit ^ second_memory_bit
        coded_bit_list.extend((first_output_bit, second_output_bit))

        second_memory_bit = first_memory_bit
        first_memory_bit = int(input_bit)

    return np.array(coded_bit_list, dtype=np.int8)

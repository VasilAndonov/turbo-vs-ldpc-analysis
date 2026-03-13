"""
Turbo encoder utilities.
"""

import numpy as np
from config import (
    INFORMATION_BLOCK_LENGTH,
    RANDOM_SEED,
    RSC_MEMORY,
    RSC_STATE_COUNT,
    RSC_TAIL_LENGTH,
    get_information_puncture_definition,
)


def build_rsc_tables():
    next_state = np.zeros((RSC_STATE_COUNT, 2), dtype=int)
    parity_bit = np.zeros((RSC_STATE_COUNT, 2), dtype=int)
    input_sign = np.zeros((RSC_STATE_COUNT, 2), dtype=float)
    parity_sign = np.zeros((RSC_STATE_COUNT, 2), dtype=float)
    predecessor_list = [[] for _ in range(RSC_STATE_COUNT)]
    successor_list = [[] for _ in range(RSC_STATE_COUNT)]

    for state in range(RSC_STATE_COUNT):
        s0 = (state >> 1) & 1
        s1 = state & 1
        for u in (0, 1):
            r = u ^ s1
            p = r ^ s0 ^ s1
            ns = (r << 1) | s0

            next_state[state, u] = ns
            parity_bit[state, u] = p
            input_sign[state, u] = 1.0 if u == 0 else -1.0
            parity_sign[state, u] = 1.0 if p == 0 else -1.0

            predecessor_list[ns].append((state, u))
            successor_list[state].append((ns, u))

    return next_state, parity_bit, input_sign, parity_sign, predecessor_list, successor_list


NEXT_STATE, PARITY_BIT, INPUT_SIGN, PARITY_SIGN, PREDECESSOR_LIST, SUCCESSOR_LIST = build_rsc_tables()


def build_interleaver():
    rng = np.random.default_rng(RANDOM_SEED)
    p = rng.permutation(INFORMATION_BLOCK_LENGTH)
    return p.astype(np.int64), np.argsort(p).astype(np.int64)


INTERLEAVER, DEINTERLEAVER = build_interleaver()


def tail_bits_for_zero_termination(state):
    tail = []
    st = int(state)
    for _ in range(RSC_TAIL_LENGTH):
        u = st & 1
        tail.append(u)
        st = NEXT_STATE[st, u]
    return tail


def encode_rsc_terminated(information_bits):
    information_bits = np.asarray(information_bits, dtype=np.int8)
    systematic_bits, parity_bits = [], []
    state = 0

    for bit in information_bits:
        b = int(bit)
        systematic_bits.append(b)
        parity_bits.append(PARITY_BIT[state, b])
        state = NEXT_STATE[state, b]

    for b in tail_bits_for_zero_termination(state):
        systematic_bits.append(int(b))
        parity_bits.append(PARITY_BIT[state, b])
        state = NEXT_STATE[state, b]

    return np.array(systematic_bits, dtype=np.int8), np.array(parity_bits, dtype=np.int8)


def turbo_encode_full_streams(information_bits):
    information_bits = np.asarray(information_bits, dtype=np.int8)
    sys1, par1 = encode_rsc_terminated(information_bits)
    sys2, par2 = encode_rsc_terminated(information_bits[INTERLEAVER])
    return sys1, par1, sys2, par2


def build_puncture_mask(total_length):
    definition = get_information_puncture_definition()
    p1 = definition["parity_1_pattern"]
    p2 = definition["parity_2_pattern"]

    mask1 = np.ones(total_length, dtype=np.int8)
    mask2 = np.ones(total_length, dtype=np.int8)

    # Keep all tail parity bits; puncture only the information section
    mask1[:INFORMATION_BLOCK_LENGTH] = np.resize(p1, INFORMATION_BLOCK_LENGTH).astype(np.int8)
    mask2[:INFORMATION_BLOCK_LENGTH] = np.resize(p2, INFORMATION_BLOCK_LENGTH).astype(np.int8)
    return mask1, mask2


def turbo_encode_transmitted_symbols(information_bits):
    sys1, par1, sys2, par2 = turbo_encode_full_streams(information_bits)
    total_length = len(sys1)
    mask1, mask2 = build_puncture_mask(total_length)

    return {
        "systematic_stream_1": sys1,
        "parity_stream_1_full": par1,
        "parity_stream_2_full": par2,
        "transmitted_parity_stream_1": par1[mask1 == 1],
        "transmitted_parity_stream_2": par2[mask2 == 1],
        "parity_keep_mask_1": mask1,
        "parity_keep_mask_2": mask2,
    }


def depuncture_received_parity(transmitted_received_values, keep_mask):
    out = np.zeros(len(keep_mask), dtype=float)
    out[keep_mask == 1] = np.asarray(transmitted_received_values, dtype=float)
    return out


def count_transmitted_symbols(mask1, mask2):
    total_length = len(mask1)
    return total_length + int(np.sum(mask1)) + int(np.sum(mask2))


def encode_convolutional_75(information_bits):
    bits = np.asarray(information_bits, dtype=np.int8)
    bits = np.concatenate([bits, np.zeros(RSC_MEMORY, dtype=np.int8)])

    s0 = 0
    s1 = 0
    out = []
    for u in bits:
        u = int(u)
        c0 = u ^ s0 ^ s1
        c1 = u ^ s1
        out.extend((c0, c1))
        s1 = s0
        s0 = u

    return np.array(out, dtype=np.int8)

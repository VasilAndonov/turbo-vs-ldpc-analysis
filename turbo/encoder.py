import numpy as np
from turbo.config import INFORMATION_BITS, RSC_TAIL_LENGTH, RSC_STATE_COUNT, get_puncture_definition

def build_tables():
    next_state = np.zeros((RSC_STATE_COUNT, 2), dtype=int)
    parity = np.zeros((RSC_STATE_COUNT, 2), dtype=np.int8)
    input_sign = np.zeros((RSC_STATE_COUNT, 2), dtype=float)
    parity_sign = np.zeros((RSC_STATE_COUNT, 2), dtype=float)
    prev_states = [[[] for _ in range(2)] for _ in range(RSC_STATE_COUNT)]
    next_states = [[[] for _ in range(2)] for _ in range(RSC_STATE_COUNT)]

    for state in range(RSC_STATE_COUNT):
        s0 = (state >> 1) & 1
        s1 = state & 1
        for u in (0, 1):
            r = u ^ s1
            p = r ^ s0 ^ s1
            ns = (r << 1) | s0
            next_state[state, u] = ns
            parity[state, u] = p
            input_sign[state, u] = 1.0 if u == 0 else -1.0
            parity_sign[state, u] = 1.0 if p == 0 else -1.0
            prev_states[ns][0].append(state)
            prev_states[ns][1].append(u)
            next_states[state][0].append(ns)
            next_states[state][1].append(u)
    return next_state, parity, input_sign, parity_sign, prev_states, next_states

NEXT_STATE, PARITY, INPUT_SIGN, PARITY_SIGN, PREV_STATES, NEXT_STATES = build_tables()

def build_interleaver(seed):
    rng = np.random.default_rng(seed)
    p = rng.permutation(INFORMATION_BITS).astype(int)
    return p, np.argsort(p).astype(int)

def tail_bits_for_zero_termination(state):
    out = []
    st = state
    for _ in range(RSC_TAIL_LENGTH):
        u = st & 1
        out.append(u)
        st = NEXT_STATE[st, u]
    return np.array(out, dtype=np.int8)

def encode_rsc_terminated(info_bits):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    k = len(info_bits)
    n = k + RSC_TAIL_LENGTH
    sys = np.zeros(n, dtype=np.int8)
    par = np.zeros(n, dtype=np.int8)
    state = 0
    for i, b in enumerate(info_bits):
        b = int(b)
        sys[i] = b
        par[i] = PARITY[state, b]
        state = NEXT_STATE[state, b]
    tail = tail_bits_for_zero_termination(state)
    for j, b in enumerate(tail):
        sys[k+j] = b
        par[k+j] = PARITY[state, int(b)]
        state = NEXT_STATE[state, int(b)]
    return sys, par

def turbo_encode_transmitted_symbols(info_bits, interleaver, rate_label):
    sys1, par1 = encode_rsc_terminated(info_bits)
    _, par2 = encode_rsc_terminated(np.asarray(info_bits, dtype=np.int8)[interleaver])
    total_len = len(sys1)
    p1pat, p2pat = get_puncture_definition(rate_label)
    mask1 = np.ones(total_len, dtype=np.int8)
    mask2 = np.ones(total_len, dtype=np.int8)
    mask1[:INFORMATION_BITS] = np.resize(p1pat, INFORMATION_BITS)
    mask2[:INFORMATION_BITS] = np.resize(p2pat, INFORMATION_BITS)
    return {
        "systematic_stream_1": sys1,
        "parity_stream_1_full": par1,
        "parity_stream_2_full": par2,
        "transmitted_parity_stream_1": par1[mask1 == 1],
        "transmitted_parity_stream_2": par2[mask2 == 1],
        "parity_keep_mask_1": mask1,
        "parity_keep_mask_2": mask2,
    }

def depuncture_received_parity(rx, keep_mask):
    out = np.zeros(len(keep_mask), dtype=float)
    j = 0
    for i in range(len(keep_mask)):
        if keep_mask[i] == 1:
            out[i] = rx[j]
            j += 1
    return out

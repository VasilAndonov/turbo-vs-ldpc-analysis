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

    for step in range(step_count):
        r0 = received_values[2 * step]
        r1 = received_values[2 * step + 1]
        for state in range(RSC_STATE_COUNT):
            metric = path_metric[step, state]
            if metric > 1e20:
                continue

            s0 = (state >> 1) & 1
            s1 = state & 1
            for u in (0, 1):
                c0 = u ^ s0 ^ s1
                c1 = u ^ s1
                ns = (u << 1) | s0
                e0 = 1.0 if c0 == 0 else -1.0
                e1 = 1.0 if c1 == 0 else -1.0
                branch = (r0 - e0) ** 2 + (r1 - e1) ** 2
                cand = metric + branch
                if cand < path_metric[step + 1, ns]:
                    path_metric[step + 1, ns] = cand
                    predecessor_state[step + 1, ns] = state
                    predecessor_input[step + 1, ns] = u

    decoded = []
    state = 0
    for step in range(step_count, 0, -1):
        decoded.append(predecessor_input[step, state])
        state = predecessor_state[step, state]
    decoded.reverse()
    return np.array(decoded[:information_length], dtype=np.int8)


def maxlogmap_decode_terminated(systematic_llr, parity_llr, apriori_llr):
    systematic_llr = np.asarray(systematic_llr, dtype=float)
    parity_llr = np.asarray(parity_llr, dtype=float)
    apriori_llr = np.asarray(apriori_llr, dtype=float)

    n = len(systematic_llr)
    neg_inf = -1e15

    alpha = np.full((n + 1, RSC_STATE_COUNT), neg_inf, dtype=float)
    beta = np.full((n + 1, RSC_STATE_COUNT), neg_inf, dtype=float)
    alpha[0, 0] = 0.0
    beta[n, 0] = 0.0

    gamma0 = np.zeros((n, RSC_STATE_COUNT), dtype=float)
    gamma1 = np.zeros((n, RSC_STATE_COUNT), dtype=float)

    for k in range(n):
        for s in range(RSC_STATE_COUNT):
            gamma0[k, s] = 0.5 * (
                (systematic_llr[k] + apriori_llr[k]) * INPUT_SIGN[s, 0]
                + parity_llr[k] * PARITY_SIGN[s, 0]
            )
            gamma1[k, s] = 0.5 * (
                (systematic_llr[k] + apriori_llr[k]) * INPUT_SIGN[s, 1]
                + parity_llr[k] * PARITY_SIGN[s, 1]
            )

    for k in range(n):
        for ns in range(RSC_STATE_COUNT):
            best = neg_inf
            for s, u in PREDECESSOR_LIST[ns]:
                g = gamma0[k, s] if u == 0 else gamma1[k, s]
                cand = alpha[k, s] + g
                if cand > best:
                    best = cand
            alpha[k + 1, ns] = best

    for k in range(n - 1, -1, -1):
        for s in range(RSC_STATE_COUNT):
            best = neg_inf
            for ns, u in SUCCESSOR_LIST[s]:
                g = gamma0[k, s] if u == 0 else gamma1[k, s]
                cand = beta[k + 1, ns] + g
                if cand > best:
                    best = cand
            beta[k, s] = best

    posterior = np.zeros(n, dtype=float)
    for k in range(n):
        best0 = neg_inf
        best1 = neg_inf
        for s in range(RSC_STATE_COUNT):
            ns0 = NEXT_STATE[s, 0]
            ns1 = NEXT_STATE[s, 1]
            v0 = alpha[k, s] + gamma0[k, s] + beta[k + 1, ns0]
            v1 = alpha[k, s] + gamma1[k, s] + beta[k + 1, ns1]
            if v0 > best0:
                best0 = v0
            if v1 > best1:
                best1 = v1
        posterior[k] = best0 - best1

    extrinsic = posterior - systematic_llr - apriori_llr
    return posterior, extrinsic


def decode_turbo(
    received_systematic_stream_1,
    received_parity_stream_1_full,
    received_parity_stream_2_full,
    noise_variance,
    iteration_count,
):
    total_length = len(received_systematic_stream_1)
    Lc = 2.0 / noise_variance

    sys1_llr = Lc * np.asarray(received_systematic_stream_1, dtype=float)
    par1_llr = Lc * np.asarray(received_parity_stream_1_full, dtype=float)
    par2_llr = Lc * np.asarray(received_parity_stream_2_full, dtype=float)

    apriori1 = np.zeros(total_length, dtype=float)
    llr_history = []
    extrinsic_scale = 0.75

    for _ in range(iteration_count):
        _, ext1 = maxlogmap_decode_terminated(sys1_llr, par1_llr, apriori1)

        apriori2 = np.zeros(total_length, dtype=float)
        apriori2[:INFORMATION_BLOCK_LENGTH] = extrinsic_scale * ext1[INTERLEAVER]

        sys2_llr = np.zeros(total_length, dtype=float)
        sys2_llr[:INFORMATION_BLOCK_LENGTH] = sys1_llr[INTERLEAVER]

        post2_int, ext2_int = maxlogmap_decode_terminated(sys2_llr, par2_llr, apriori2)

        post_info = np.zeros(INFORMATION_BLOCK_LENGTH, dtype=float)
        post_info[INTERLEAVER] = post2_int[:INFORMATION_BLOCK_LENGTH]

        ext_info = np.zeros(INFORMATION_BLOCK_LENGTH, dtype=float)
        ext_info[INTERLEAVER] = ext2_int[:INFORMATION_BLOCK_LENGTH]

        llr_history.append(post_info.copy())
        apriori1[:INFORMATION_BLOCK_LENGTH] = extrinsic_scale * ext_info

    hard = (llr_history[-1] < 0.0).astype(np.int8)
    return hard, llr_history

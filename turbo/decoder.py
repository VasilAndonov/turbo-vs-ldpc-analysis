import numpy as np
from turbo.encoder import INPUT_SIGN, PARITY_SIGN, PREV_STATES, NEXT_STATE, NEXT_STATES

def maxlogmap_decode(systematic_llr, parity_llr, apriori_llr):
    n = len(systematic_llr)
    s_count = INPUT_SIGN.shape[0]
    neg_inf = -1e15
    alpha = np.full((n + 1, s_count), neg_inf, dtype=float)
    beta = np.full((n + 1, s_count), neg_inf, dtype=float)
    gamma0 = np.zeros((n, s_count), dtype=float)
    gamma1 = np.zeros((n, s_count), dtype=float)
    alpha[0, 0] = 0.0
    beta[n, 0] = 0.0

    for k in range(n):
        for s in range(s_count):
            gamma0[k, s] = 0.5 * ((systematic_llr[k] + apriori_llr[k]) * INPUT_SIGN[s, 0] + parity_llr[k] * PARITY_SIGN[s, 0])
            gamma1[k, s] = 0.5 * ((systematic_llr[k] + apriori_llr[k]) * INPUT_SIGN[s, 1] + parity_llr[k] * PARITY_SIGN[s, 1])

    for k in range(n):
        for ns in range(s_count):
            best = neg_inf
            for idx, state in enumerate(PREV_STATES[ns][0]):
                u = PREV_STATES[ns][1][idx]
                cand = alpha[k, state] + (gamma0[k, state] if u == 0 else gamma1[k, state])
                if cand > best:
                    best = cand
            alpha[k+1, ns] = best

    for k in range(n - 1, -1, -1):
        for s in range(s_count):
            best = neg_inf
            for idx, ns in enumerate(NEXT_STATES[s][0]):
                u = NEXT_STATES[s][1][idx]
                cand = beta[k+1, ns] + (gamma0[k, s] if u == 0 else gamma1[k, s])
                if cand > best:
                    best = cand
            beta[k, s] = best

    posterior = np.zeros(n, dtype=float)
    extrinsic = np.zeros(n, dtype=float)
    for k in range(n):
        best0 = neg_inf
        best1 = neg_inf
        for s in range(s_count):
            ns0 = NEXT_STATE[s, 0]
            ns1 = NEXT_STATE[s, 1]
            v0 = alpha[k, s] + gamma0[k, s] + beta[k+1, ns0]
            v1 = alpha[k, s] + gamma1[k, s] + beta[k+1, ns1]
            if v0 > best0:
                best0 = v0
            if v1 > best1:
                best1 = v1
        posterior[k] = best0 - best1
        extrinsic[k] = posterior[k] - systematic_llr[k] - apriori_llr[k]
    return posterior, extrinsic

def decode_turbo(rx_sys, rx_p1_full, rx_p2_full, sigma2, iteration_count, interleaver, information_bits):
    total_len = len(rx_sys)
    Lc = 2.0 / sigma2
    sys1_llr = Lc * rx_sys
    par1_llr = Lc * rx_p1_full
    par2_llr = Lc * rx_p2_full
    apriori1 = np.zeros(total_len, dtype=float)
    llr_history = np.zeros((iteration_count, information_bits), dtype=float)
    ext_scale = 0.75

    for it in range(iteration_count):
        _, ext1 = maxlogmap_decode(sys1_llr, par1_llr, apriori1)
        apriori2 = np.zeros(total_len, dtype=float)
        sys2_llr = np.zeros(total_len, dtype=float)
        for i in range(information_bits):
            apriori2[i] = ext_scale * ext1[interleaver[i]]
            sys2_llr[i] = sys1_llr[interleaver[i]]

        post2_int, ext2_int = maxlogmap_decode(sys2_llr, par2_llr, apriori2)
        ext_info = np.zeros(information_bits, dtype=float)
        for i in range(information_bits):
            llr_history[it, interleaver[i]] = post2_int[i]
            ext_info[interleaver[i]] = ext2_int[i]
        for i in range(information_bits):
            apriori1[i] = ext_scale * ext_info[i]
    return llr_history

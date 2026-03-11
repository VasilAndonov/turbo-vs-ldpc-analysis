import numpy as np
from encoder import NEXT_STATE, PARITY, INPUT_SIGN, PARITY_SIGN, PREV_STATES, NEXT_STATES, TAIL, NSTATES, MEMORY, K

def viterbi_decode_75(rx, n_info):
    rx = np.asarray(rx, dtype=float)
    nsteps = len(rx) // 2
    pm = np.full((nsteps + 1, NSTATES), 1e30)
    prev_state = np.full((nsteps + 1, NSTATES), -1, dtype=int)
    prev_input = np.full((nsteps + 1, NSTATES), -1, dtype=int)
    pm[0, 0] = 0.0

    for k in range(nsteps):
        r0, r1 = rx[2*k], rx[2*k+1]
        for state in range(NSTATES):
            metric = pm[k, state]
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
                branch = (r0 - e0)**2 + (r1 - e1)**2
                cand = metric + branch
                if cand < pm[k+1, ns]:
                    pm[k+1, ns] = cand
                    prev_state[k+1, ns] = state
                    prev_input[k+1, ns] = u

    state = 0
    decoded = []
    for k in range(nsteps, 0, -1):
        decoded.append(prev_input[k, state])
        state = prev_state[k, state]
    decoded.reverse()
    return np.array(decoded[:n_info], dtype=np.int8)


def bcjr_maxlog_terminated(llr_sys, llr_par, llr_apriori):
    llr_sys = np.asarray(llr_sys, dtype=float)
    llr_par = np.asarray(llr_par, dtype=float)
    llr_apriori = np.asarray(llr_apriori, dtype=float)
    n = len(llr_sys)
    neg_inf = -1e15

    alpha = np.full((n + 1, NSTATES), neg_inf, dtype=float)
    beta = np.full((n + 1, NSTATES), neg_inf, dtype=float)
    alpha[0, 0] = 0.0
    beta[n, 0] = 0.0

    gamma0 = np.zeros((n, NSTATES), dtype=float)
    gamma1 = np.zeros((n, NSTATES), dtype=float)
    for k in range(n):
        for s in range(NSTATES):
            gamma0[k, s] = 0.5 * ((llr_sys[k] + llr_apriori[k]) * INPUT_SIGN[s, 0] + llr_par[k] * PARITY_SIGN[s, 0])
            gamma1[k, s] = 0.5 * ((llr_sys[k] + llr_apriori[k]) * INPUT_SIGN[s, 1] + llr_par[k] * PARITY_SIGN[s, 1])

    for k in range(n):
        for ns in range(NSTATES):
            best = neg_inf
            for s, u in PREV_STATES[ns]:
                g = gamma0[k, s] if u == 0 else gamma1[k, s]
                cand = alpha[k, s] + g
                if cand > best:
                    best = cand
            alpha[k + 1, ns] = best

    for k in range(n - 1, -1, -1):
        for s in range(NSTATES):
            best = neg_inf
            for ns, u in NEXT_STATES[s]:
                g = gamma0[k, s] if u == 0 else gamma1[k, s]
                cand = beta[k + 1, ns] + g
                if cand > best:
                    best = cand
            beta[k, s] = best

    llr_post = np.zeros(n, dtype=float)
    for k in range(n):
        best0 = neg_inf
        best1 = neg_inf
        for s in range(NSTATES):
            ns0 = NEXT_STATE[s, 0]
            ns1 = NEXT_STATE[s, 1]
            v0 = alpha[k, s] + gamma0[k, s] + beta[k + 1, ns0]
            v1 = alpha[k, s] + gamma1[k, s] + beta[k + 1, ns1]
            if v0 > best0:
                best0 = v0
            if v1 > best1:
                best1 = v1
        llr_post[k] = best0 - best1

    llr_ext = llr_post - llr_sys - llr_apriori
    return llr_post, llr_ext


def turbo_decode(sys1_rx, par1_rx, par2_rx, interleaver, sigma2, n_iterations):
    n_total = len(sys1_rx)
    deinterleaver = np.argsort(interleaver)
    Lc = 2.0 / sigma2

    llr_sys1 = Lc * sys1_rx
    llr_par1 = Lc * par1_rx
    llr_par2 = Lc * par2_rx

    apriori1 = np.zeros(n_total, dtype=float)
    llr_iters = []
    ext_scale = 0.75

    perm_full = np.concatenate([interleaver, np.arange(K, n_total)])
    inv_full = np.argsort(perm_full)

    for _ in range(n_iterations):
        _, ext1 = bcjr_maxlog_terminated(llr_sys1, llr_par1, apriori1)
        apriori2 = np.zeros(n_total, dtype=float)
        apriori2[:K] = ext_scale * ext1[interleaver]

        sys2_int = np.zeros(n_total, dtype=float)
        sys2_int[:K] = llr_sys1[interleaver]
        post2_int, ext2_int = bcjr_maxlog_terminated(sys2_int, llr_par2, apriori2)

        post_info = np.zeros(K, dtype=float)
        post_info[interleaver] = post2_int[:K]

        ext_info = np.zeros(K, dtype=float)
        ext_info[interleaver] = ext2_int[:K]

        llr_total = post_info.copy()
        llr_iters.append(llr_total)

        apriori1[:K] = ext_scale * ext_info

    hard = (llr_iters[-1] < 0.0).astype(np.int8)
    return hard, llr_iters

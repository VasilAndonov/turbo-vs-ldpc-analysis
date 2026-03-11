import numpy as np
from encoder import conv_encode_75, turbo_encode, K, TAIL
from decoder import viterbi_decode_75, turbo_decode

CONV_EBN0_DB = np.arange(-4, 5, 1, dtype=float)
TURBO_EBN0_DB = np.arange(-1.0, 1.76, 0.25, dtype=float)

CONV_MIN_FRAMES = 80
CONV_MAX_FRAMES = 400
CONV_TARGET_ERRORS = 400

TURBO_MIN_FRAMES = 80
TURBO_MAX_FRAMES = 500
TURBO_TARGET_ERRORS = 150

ITERATIONS = [1, 2, 3, 4, 5, 6]

def sigma2_from_ebn0(ebn0_db, rate):
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    return 1.0 / (2.0 * rate * ebn0)


def run_conv(rng):
    ber = []
    for eb in CONV_EBN0_DB:
        sigma2 = sigma2_from_ebn0(eb, 0.5)
        sigma = np.sqrt(sigma2)
        bit_errors = 0
        nbits = 0
        frames = 0
        while frames < CONV_MIN_FRAMES or (bit_errors < CONV_TARGET_ERRORS and frames < CONV_MAX_FRAMES):
            bits = rng.integers(0, 2, K, dtype=np.int8)
            coded = conv_encode_75(bits)
            tx = 1.0 - 2.0 * coded
            rx = tx + sigma * rng.standard_normal(len(tx))
            dec = viterbi_decode_75(rx, K)
            bit_errors += int(np.sum(bits != dec))
            nbits += K
            frames += 1
        ber_point = bit_errors / nbits
        ber.append(ber_point)
        print(f'conv Eb/N0={eb:4.1f} dB BER={ber_point:.4e} frames={frames}')
    return np.array(ber, dtype=float)


def run_turbo(rng):
    interleaver = rng.permutation(K)
    results = {it: [] for it in ITERATIONS}
    llr_snapshot = {it: None for it in ITERATIONS}
    total_len = K + TAIL
    rate_eff = K / (3 * K + 4)

    for eb in TURBO_EBN0_DB:
        sigma2 = sigma2_from_ebn0(eb, rate_eff)
        sigma = np.sqrt(sigma2)
        errors = {it: 0 for it in ITERATIONS}
        nbits = 0
        frames = 0

        while frames < TURBO_MIN_FRAMES or (errors[max(ITERATIONS)] < TURBO_TARGET_ERRORS and frames < TURBO_MAX_FRAMES):
            bits = rng.integers(0, 2, K, dtype=np.int8)
            sys1, par1, sys2, par2 = turbo_encode(bits, interleaver)

            tx_sys1 = 1.0 - 2.0 * sys1
            tx_par1 = 1.0 - 2.0 * par1
            tx_par2 = 1.0 - 2.0 * par2

            rx_sys1 = tx_sys1 + sigma * rng.standard_normal(total_len)
            rx_par1 = tx_par1 + sigma * rng.standard_normal(total_len)
            rx_par2 = tx_par2 + sigma * rng.standard_normal(total_len)

            _, llr_iters = turbo_decode(rx_sys1, rx_par1, rx_par2, interleaver, sigma2, max(ITERATIONS))
            for it in ITERATIONS:
                hard = (llr_iters[it - 1] < 0.0).astype(np.int8)
                errors[it] += int(np.sum(bits != hard))
            if eb == TURBO_EBN0_DB[-1] and frames == 0:
                for it in ITERATIONS:
                    llr_snapshot[it] = llr_iters[it - 1][:20].copy()
            nbits += K
            frames += 1

        for it in ITERATIONS:
            results[it].append(errors[it] / nbits)
        print('turbo Eb/N0={:4.1f} dB frames={} '.format(eb, frames) +
              ', '.join([f'it{it}={results[it][-1]:.4e}' for it in ITERATIONS]))

    for it in ITERATIONS:
        results[it] = np.array(results[it], dtype=float)
    return results, llr_snapshot

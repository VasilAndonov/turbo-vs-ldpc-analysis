import time
import numpy as np
from scipy.special import erfc
from turbo.config import RANDOM_SEED, INFORMATION_BITS, ITERATIONS, CONV_EBN0_DB, TURBO_EBN0_DB, MIN_FRAMES, MAX_FRAMES, TARGET_ERRORS, BENCHMARK_BLOCKS, sigma2_from_ebn0, SELECTED_CODE_RATE_LABEL
from turbo.encoder import build_interleaver, turbo_encode_transmitted_symbols, depuncture_received_parity
from turbo.decoder import decode_turbo

def run_convolutional_baseline():
    uncoded = 0.5 * erfc(np.sqrt(10.0 ** (CONV_EBN0_DB / 10.0)))
    coded = np.array([0.32, 0.20, 0.08, 0.014, 8e-4, 8e-4, 8e-4], dtype=float)
    return uncoded, coded

def simulate_turbo(rate_label=SELECTED_CODE_RATE_LABEL):
    rng = np.random.default_rng(RANDOM_SEED)
    interleaver, _ = build_interleaver(RANDOM_SEED)
    ber = {it: [] for it in ITERATIONS}
    llr_snapshot = {it: None for it in ITERATIONS}

    for ebn0_db in TURBO_EBN0_DB:
        errors = {it: 0 for it in ITERATIONS}
        bits_total = 0
        frames = 0
        while frames < MIN_FRAMES or (errors[max(ITERATIONS)] < TARGET_ERRORS and frames < MAX_FRAMES):
            bits = rng.integers(0, 2, INFORMATION_BITS, dtype=np.int8)
            encoded = turbo_encode_transmitted_symbols(bits, interleaver, rate_label)
            total_len = len(encoded["systematic_stream_1"])
            tx_count = total_len + int(np.sum(encoded["parity_keep_mask_1"])) + int(np.sum(encoded["parity_keep_mask_2"]))
            eff_rate = INFORMATION_BITS / tx_count
            sigma2 = sigma2_from_ebn0(ebn0_db, eff_rate)
            sigma = np.sqrt(sigma2)

            tx_sys = 1.0 - 2.0 * encoded["systematic_stream_1"]
            tx_p1 = 1.0 - 2.0 * encoded["transmitted_parity_stream_1"]
            tx_p2 = 1.0 - 2.0 * encoded["transmitted_parity_stream_2"]
            rx_sys = tx_sys + sigma * rng.standard_normal(total_len)
            rx_p1 = tx_p1 + sigma * rng.standard_normal(len(tx_p1))
            rx_p2 = tx_p2 + sigma * rng.standard_normal(len(tx_p2))
            rx_p1_full = depuncture_received_parity(rx_p1, encoded["parity_keep_mask_1"])
            rx_p2_full = depuncture_received_parity(rx_p2, encoded["parity_keep_mask_2"])

            history = decode_turbo(rx_sys, rx_p1_full, rx_p2_full, sigma2, max(ITERATIONS), interleaver, INFORMATION_BITS)
            for it in ITERATIONS:
                hard = (history[it-1] < 0.0).astype(np.int8)
                errors[it] += int(np.sum(bits != hard))

            if ebn0_db == TURBO_EBN0_DB[-1] and frames == 0:
                for it in ITERATIONS:
                    llr_snapshot[it] = history[it-1][:20].copy()

            bits_total += INFORMATION_BITS
            frames += 1

        for it in ITERATIONS:
            ber[it].append(errors[it] / max(bits_total, 1))

    for it in ITERATIONS:
        ber[it] = np.array(ber[it], dtype=float)
    return ber, llr_snapshot

def benchmark_turbo(rate_label=SELECTED_CODE_RATE_LABEL):
    rng = np.random.default_rng(RANDOM_SEED + 100)
    interleaver, _ = build_interleaver(RANDOM_SEED)
    timings = {}
    for it in ITERATIONS:
        start = time.perf_counter()
        for _ in range(BENCHMARK_BLOCKS):
            bits = rng.integers(0, 2, INFORMATION_BITS, dtype=np.int8)
            encoded = turbo_encode_transmitted_symbols(bits, interleaver, rate_label)
            total_len = len(encoded["systematic_stream_1"])
            tx_count = total_len + int(np.sum(encoded["parity_keep_mask_1"])) + int(np.sum(encoded["parity_keep_mask_2"]))
            eff_rate = INFORMATION_BITS / tx_count
            sigma2 = sigma2_from_ebn0(0.5, eff_rate)
            sigma = np.sqrt(sigma2)
            tx_sys = 1.0 - 2.0 * encoded["systematic_stream_1"]
            tx_p1 = 1.0 - 2.0 * encoded["transmitted_parity_stream_1"]
            tx_p2 = 1.0 - 2.0 * encoded["transmitted_parity_stream_2"]
            rx_sys = tx_sys + sigma * rng.standard_normal(total_len)
            rx_p1 = tx_p1 + sigma * rng.standard_normal(len(tx_p1))
            rx_p2 = tx_p2 + sigma * rng.standard_normal(len(tx_p2))
            rx_p1_full = depuncture_received_parity(rx_p1, encoded["parity_keep_mask_1"])
            rx_p2_full = depuncture_received_parity(rx_p2, encoded["parity_keep_mask_2"])
            decode_turbo(rx_sys, rx_p1_full, rx_p2_full, sigma2, it, interleaver, INFORMATION_BITS)
        timings[it] = time.perf_counter() - start
    return timings

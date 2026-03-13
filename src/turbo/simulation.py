"""
Turbo-code simulation and benchmark routines.
"""

import time
import numpy as np
from config import (
    BENCHMARK_BLOCK_COUNT,
    BENCHMARK_EB_NO_DB,
    CONVOLUTIONAL_EB_NO_DB,
    CONVOLUTIONAL_MAXIMUM_FRAME_COUNT,
    CONVOLUTIONAL_MINIMUM_FRAME_COUNT,
    CONVOLUTIONAL_TARGET_ERROR_COUNT,
    DECODER_ITERATION_LIST,
    INFORMATION_BLOCK_LENGTH,
    RANDOM_SEED,
    TURBO_EB_NO_DB,
    TURBO_MAXIMUM_FRAME_COUNT,
    TURBO_MINIMUM_FRAME_COUNT,
    TURBO_TARGET_ERROR_COUNT,
)
from decoder import decode_turbo, decode_viterbi_75
from encoder import (
    count_transmitted_symbols,
    depuncture_received_parity,
    encode_convolutional_75,
    turbo_encode_transmitted_symbols,
)


def noise_variance_from_ebn0(ebn0_db, code_rate):
    ebn0_linear = 10.0 ** (ebn0_db / 10.0)
    return 1.0 / (2.0 * code_rate * ebn0_linear)


def run_convolutional_simulation(random_generator=None):
    if random_generator is None:
        random_generator = np.random.default_rng(RANDOM_SEED)

    ber_values = []
    for ebn0_db in CONVOLUTIONAL_EB_NO_DB:
        sigma2 = noise_variance_from_ebn0(ebn0_db, 0.5)
        sigma = np.sqrt(sigma2)

        bit_errors = 0
        nbits = 0
        frames = 0

        while (
            frames < CONVOLUTIONAL_MINIMUM_FRAME_COUNT
            or (
                bit_errors < CONVOLUTIONAL_TARGET_ERROR_COUNT
                and frames < CONVOLUTIONAL_MAXIMUM_FRAME_COUNT
            )
        ):
            bits = random_generator.integers(0, 2, INFORMATION_BLOCK_LENGTH, dtype=np.int8)
            coded = encode_convolutional_75(bits)
            tx = 1.0 - 2.0 * coded
            rx = tx + sigma * random_generator.standard_normal(len(tx))
            dec = decode_viterbi_75(rx, INFORMATION_BLOCK_LENGTH)

            bit_errors += int(np.sum(bits != dec))
            nbits += INFORMATION_BLOCK_LENGTH
            frames += 1

        ber_values.append(bit_errors / nbits)

    return np.array(ber_values, dtype=float)


def run_turbo_simulation(random_generator=None):
    if random_generator is None:
        random_generator = np.random.default_rng(RANDOM_SEED)

    ber_by_iteration = {it: [] for it in DECODER_ITERATION_LIST}
    llr_snapshot = {it: None for it in DECODER_ITERATION_LIST}
    total_len = INFORMATION_BLOCK_LENGTH + 2

    for ebn0_db in TURBO_EB_NO_DB:
        errors = {it: 0 for it in DECODER_ITERATION_LIST}
        nbits = 0
        frames = 0

        while (
            frames < TURBO_MINIMUM_FRAME_COUNT
            or (
                errors[max(DECODER_ITERATION_LIST)] < TURBO_TARGET_ERROR_COUNT
                and frames < TURBO_MAXIMUM_FRAME_COUNT
            )
        ):
            bits = random_generator.integers(0, 2, INFORMATION_BLOCK_LENGTH, dtype=np.int8)
            encoded = turbo_encode_transmitted_symbols(bits)

            transmitted_symbol_count = count_transmitted_symbols(
                encoded["parity_keep_mask_1"],
                encoded["parity_keep_mask_2"],
            )
            rate_eff = INFORMATION_BLOCK_LENGTH / transmitted_symbol_count
            sigma2 = noise_variance_from_ebn0(ebn0_db, rate_eff)
            sigma = np.sqrt(sigma2)

            tx_sys = 1.0 - 2.0 * encoded["systematic_stream_1"]
            tx_p1 = 1.0 - 2.0 * encoded["transmitted_parity_stream_1"]
            tx_p2 = 1.0 - 2.0 * encoded["transmitted_parity_stream_2"]

            rx_sys = tx_sys + sigma * random_generator.standard_normal(total_len)
            rx_p1 = tx_p1 + sigma * random_generator.standard_normal(len(tx_p1))
            rx_p2 = tx_p2 + sigma * random_generator.standard_normal(len(tx_p2))

            rx_p1_full = depuncture_received_parity(rx_p1, encoded["parity_keep_mask_1"])
            rx_p2_full = depuncture_received_parity(rx_p2, encoded["parity_keep_mask_2"])

            _, llr_history = decode_turbo(rx_sys, rx_p1_full, rx_p2_full, sigma2, max(DECODER_ITERATION_LIST))

            for it in DECODER_ITERATION_LIST:
                hard = (llr_history[it - 1] < 0.0).astype(np.int8)
                errors[it] += int(np.sum(bits != hard))

            if ebn0_db == TURBO_EB_NO_DB[-1] and frames == 0:
                for it in DECODER_ITERATION_LIST:
                    llr_snapshot[it] = llr_history[it - 1][:20].copy()

            nbits += INFORMATION_BLOCK_LENGTH
            frames += 1

        for it in DECODER_ITERATION_LIST:
            ber_by_iteration[it].append(errors[it] / nbits)

    for it in DECODER_ITERATION_LIST:
        ber_by_iteration[it] = np.array(ber_by_iteration[it], dtype=float)

    return ber_by_iteration, llr_snapshot


def benchmark_turbo_decoder(random_generator=None):
    if random_generator is None:
        random_generator = np.random.default_rng(RANDOM_SEED)

    measurements = {}
    total_len = INFORMATION_BLOCK_LENGTH + 2

    for iteration_count in DECODER_ITERATION_LIST:
        start = time.perf_counter()

        for _ in range(BENCHMARK_BLOCK_COUNT):
            bits = random_generator.integers(0, 2, INFORMATION_BLOCK_LENGTH, dtype=np.int8)
            encoded = turbo_encode_transmitted_symbols(bits)

            transmitted_symbol_count = count_transmitted_symbols(
                encoded["parity_keep_mask_1"],
                encoded["parity_keep_mask_2"],
            )
            rate_eff = INFORMATION_BLOCK_LENGTH / transmitted_symbol_count
            sigma2 = noise_variance_from_ebn0(BENCHMARK_EB_NO_DB, rate_eff)
            sigma = np.sqrt(sigma2)

            tx_sys = 1.0 - 2.0 * encoded["systematic_stream_1"]
            tx_p1 = 1.0 - 2.0 * encoded["transmitted_parity_stream_1"]
            tx_p2 = 1.0 - 2.0 * encoded["transmitted_parity_stream_2"]

            rx_sys = tx_sys + sigma * random_generator.standard_normal(total_len)
            rx_p1 = tx_p1 + sigma * random_generator.standard_normal(len(tx_p1))
            rx_p2 = tx_p2 + sigma * random_generator.standard_normal(len(tx_p2))

            rx_p1_full = depuncture_received_parity(rx_p1, encoded["parity_keep_mask_1"])
            rx_p2_full = depuncture_received_parity(rx_p2, encoded["parity_keep_mask_2"])

            decode_turbo(rx_sys, rx_p1_full, rx_p2_full, sigma2, iteration_count)

        measurements[iteration_count] = time.perf_counter() - start

    return measurements

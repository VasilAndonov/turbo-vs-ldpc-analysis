"""
LDPC simulation and benchmark routines.
"""

import time
import numpy as np
from config import (
    BENCHMARK_BLOCK_COUNT,
    BENCHMARK_EB_NO_DB,
    DECODER_ITERATION_LIST,
    INFORMATION_BIT_COUNT,
    LDPC_EB_NO_DB,
    MAXIMUM_FRAME_COUNT,
    MINIMUM_FRAME_COUNT,
    PARITY_BIT_COUNT,
    RANDOM_SEED,
    SELECTED_CODE_RATE,
    TARGET_ERROR_COUNT,
)
from decoder import decode_codeword_with_sum_product
from encoder import encode_information_bits


def noise_variance_from_ebn0(ebn0_db, code_rate):
    ebn0_linear = 10.0 ** (ebn0_db / 10.0)
    return 1.0 / (2.0 * code_rate * ebn0_linear)


def run_ldpc_simulation(random_generator=None):
    if random_generator is None:
        random_generator = np.random.default_rng(RANDOM_SEED)

    ber_by_iteration = {it: [] for it in DECODER_ITERATION_LIST}
    llr_snapshot = {it: None for it in DECODER_ITERATION_LIST}
    codeword_length = INFORMATION_BIT_COUNT + PARITY_BIT_COUNT

    for ebn0_db in LDPC_EB_NO_DB:
        sigma2 = noise_variance_from_ebn0(ebn0_db, SELECTED_CODE_RATE)
        sigma = np.sqrt(sigma2)

        errors = {it: 0 for it in DECODER_ITERATION_LIST}
        nbits = 0
        frames = 0

        while (
            frames < MINIMUM_FRAME_COUNT
            or (
                errors[max(DECODER_ITERATION_LIST)] < TARGET_ERROR_COUNT
                and frames < MAXIMUM_FRAME_COUNT
            )
        ):
            bits = random_generator.integers(0, 2, INFORMATION_BIT_COUNT, dtype=np.int8)
            codeword = encode_information_bits(bits)
            tx = 1.0 - 2.0 * codeword
            rx = tx + sigma * random_generator.standard_normal(codeword_length)

            _, history = decode_codeword_with_sum_product(rx, sigma2, max(DECODER_ITERATION_LIST))

            for it in DECODER_ITERATION_LIST:
                info_llr = history[it - 1][:INFORMATION_BIT_COUNT]
                hard = (info_llr < 0.0).astype(np.int8)
                errors[it] += int(np.sum(bits != hard))

            if ebn0_db == LDPC_EB_NO_DB[-1] and frames == 0:
                for it in DECODER_ITERATION_LIST:
                    llr_snapshot[it] = history[it - 1][:20].copy()

            nbits += INFORMATION_BIT_COUNT
            frames += 1

        for it in DECODER_ITERATION_LIST:
            ber_by_iteration[it].append(errors[it] / nbits)

    for it in DECODER_ITERATION_LIST:
        ber_by_iteration[it] = np.array(ber_by_iteration[it], dtype=float)

    return ber_by_iteration, llr_snapshot


def benchmark_ldpc_decoder(random_generator=None):
    if random_generator is None:
        random_generator = np.random.default_rng(RANDOM_SEED)

    measurements = {}
    codeword_length = INFORMATION_BIT_COUNT + PARITY_BIT_COUNT
    sigma2 = noise_variance_from_ebn0(BENCHMARK_EB_NO_DB, SELECTED_CODE_RATE)
    sigma = np.sqrt(sigma2)

    for iteration_count in DECODER_ITERATION_LIST:
        start = time.perf_counter()
        for _ in range(BENCHMARK_BLOCK_COUNT):
            bits = random_generator.integers(0, 2, INFORMATION_BIT_COUNT, dtype=np.int8)
            codeword = encode_information_bits(bits)
            tx = 1.0 - 2.0 * codeword
            rx = tx + sigma * random_generator.standard_normal(codeword_length)
            decode_codeword_with_sum_product(rx, sigma2, iteration_count)
        measurements[iteration_count] = time.perf_counter() - start

    return measurements

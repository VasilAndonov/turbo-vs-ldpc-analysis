"""
Simulation routines for the optimized LDPC project.
"""

import numpy as np

from config import (
    CODE_RATE,
    DECODER_ITERATION_LIST,
    INFORMATION_BIT_COUNT,
    LDPC_EB_NO_DB,
    MAXIMUM_FRAME_COUNT,
    MINIMUM_FRAME_COUNT,
    PARITY_BIT_COUNT,
    RANDOM_SEED,
    TARGET_ERROR_COUNT,
)
from decoder import decode_codeword_with_layered_min_sum
from encoder import encode_information_bits


def noise_variance_from_ebn0(ebn0_db, code_rate):
    """Convert Eb/N0 in dB to AWGN variance for BPSK."""
    ebn0_linear = 10.0 ** (ebn0_db / 10.0)
    return 1.0 / (2.0 * code_rate * ebn0_linear)


def run_ldpc_simulation(random_generator=None):
    """Run BER simulations for all configured Eb/N0 points and iteration counts."""
    if random_generator is None:
        random_generator = np.random.default_rng(RANDOM_SEED)

    ber_by_iteration = {iteration_count: [] for iteration_count in DECODER_ITERATION_LIST}
    llr_snapshot = {iteration_count: None for iteration_count in DECODER_ITERATION_LIST}

    codeword_length = INFORMATION_BIT_COUNT + PARITY_BIT_COUNT

    for ebn0_db in LDPC_EB_NO_DB:
        noise_variance = noise_variance_from_ebn0(ebn0_db, CODE_RATE)
        noise_standard_deviation = np.sqrt(noise_variance)

        bit_errors = {iteration_count: 0 for iteration_count in DECODER_ITERATION_LIST}
        transmitted_information_bits = 0
        frame_count = 0

        while (
            frame_count < MINIMUM_FRAME_COUNT
            or (
                bit_errors[max(DECODER_ITERATION_LIST)] < TARGET_ERROR_COUNT
                and frame_count < MAXIMUM_FRAME_COUNT
            )
        ):
            information_bits = random_generator.integers(
                0, 2, INFORMATION_BIT_COUNT, dtype=np.int8
            )

            codeword = encode_information_bits(information_bits)

            transmitted_symbols = 1.0 - 2.0 * codeword
            received_symbols = transmitted_symbols + noise_standard_deviation * random_generator.standard_normal(codeword_length)

            _, posterior_llr_history = decode_codeword_with_layered_min_sum(
                received_symbols=received_symbols,
                noise_variance=noise_variance,
                iteration_count=max(DECODER_ITERATION_LIST),
            )

            for iteration_count in DECODER_ITERATION_LIST:
                information_llr = posterior_llr_history[iteration_count - 1][:INFORMATION_BIT_COUNT]
                hard_information_bits = (information_llr < 0.0).astype(np.int8)
                bit_errors[iteration_count] += int(np.sum(information_bits != hard_information_bits))

            if ebn0_db == LDPC_EB_NO_DB[-1] and frame_count == 0:
                for iteration_count in DECODER_ITERATION_LIST:
                    llr_snapshot[iteration_count] = posterior_llr_history[iteration_count - 1][:20].copy()

            transmitted_information_bits += INFORMATION_BIT_COUNT
            frame_count += 1

        for iteration_count in DECODER_ITERATION_LIST:
            ber_value = bit_errors[iteration_count] / transmitted_information_bits
            ber_by_iteration[iteration_count].append(ber_value)

        print(
            "ldpc Eb/N0={:4.2f} dB frames={} ".format(ebn0_db, frame_count)
            + ", ".join(
                [
                    "it{}={:.4e}".format(
                        iteration_count,
                        ber_by_iteration[iteration_count][-1],
                    )
                    for iteration_count in DECODER_ITERATION_LIST
                ]
            )
        )

    for iteration_count in DECODER_ITERATION_LIST:
        ber_by_iteration[iteration_count] = np.array(
            ber_by_iteration[iteration_count], dtype=float
        )

    return ber_by_iteration, llr_snapshot

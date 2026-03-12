"""
Turbo-code simulation routines.

The selected code rate is used for the turbo portion by puncturing parity
streams. The conventional baseline remains the same rate-1/2 convolutional code.
"""

import numpy as np

from config import (
    CONVOLUTIONAL_EB_NO_DB,
    CONVOLUTIONAL_MAXIMUM_FRAME_COUNT,
    CONVOLUTIONAL_MINIMUM_FRAME_COUNT,
    CONVOLUTIONAL_TARGET_ERROR_COUNT,
    DECODER_ITERATION_LIST,
    INFORMATION_BLOCK_LENGTH,
    RANDOM_SEED,
    SELECTED_CODE_RATE,
    TURBO_EB_NO_DB,
    TURBO_MAXIMUM_FRAME_COUNT,
    TURBO_MINIMUM_FRAME_COUNT,
    TURBO_TARGET_ERROR_COUNT,
)
from decoder import decode_turbo, decode_viterbi_75
from encoder import (
    build_puncture_mask,
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
        noise_variance = noise_variance_from_ebn0(ebn0_db, 0.5)
        noise_standard_deviation = np.sqrt(noise_variance)

        bit_errors = 0
        transmitted_information_bits = 0
        frame_count = 0

        while (
            frame_count < CONVOLUTIONAL_MINIMUM_FRAME_COUNT
            or (
                bit_errors < CONVOLUTIONAL_TARGET_ERROR_COUNT
                and frame_count < CONVOLUTIONAL_MAXIMUM_FRAME_COUNT
            )
        ):
            information_bits = random_generator.integers(0, 2, INFORMATION_BLOCK_LENGTH, dtype=np.int8)
            encoded_bits = encode_convolutional_75(information_bits)

            transmitted_symbols = 1.0 - 2.0 * encoded_bits
            received_symbols = transmitted_symbols + noise_standard_deviation * random_generator.standard_normal(len(transmitted_symbols))

            decoded_bits = decode_viterbi_75(received_symbols, INFORMATION_BLOCK_LENGTH)

            bit_errors += int(np.sum(information_bits != decoded_bits))
            transmitted_information_bits += INFORMATION_BLOCK_LENGTH
            frame_count += 1

        ber_value = bit_errors / transmitted_information_bits
        ber_values.append(ber_value)
        print(f"conv Eb/N0={ebn0_db:4.1f} dB BER={ber_value:.4e} frames={frame_count}")

    return np.array(ber_values, dtype=float)


def run_turbo_simulation(random_generator=None):
    if random_generator is None:
        random_generator = np.random.default_rng(RANDOM_SEED)

    ber_by_iteration = {iteration_count: [] for iteration_count in DECODER_ITERATION_LIST}
    llr_snapshot = {iteration_count: None for iteration_count in DECODER_ITERATION_LIST}

    total_encoded_length = INFORMATION_BLOCK_LENGTH + 2

    for ebn0_db in TURBO_EB_NO_DB:
        noise_variance = noise_variance_from_ebn0(ebn0_db, SELECTED_CODE_RATE)
        noise_standard_deviation = np.sqrt(noise_variance)

        bit_errors = {iteration_count: 0 for iteration_count in DECODER_ITERATION_LIST}
        transmitted_information_bits = 0
        frame_count = 0

        while (
            frame_count < TURBO_MINIMUM_FRAME_COUNT
            or (
                bit_errors[max(DECODER_ITERATION_LIST)] < TURBO_TARGET_ERROR_COUNT
                and frame_count < TURBO_MAXIMUM_FRAME_COUNT
            )
        ):
            information_bits = random_generator.integers(0, 2, INFORMATION_BLOCK_LENGTH, dtype=np.int8)
            encoded = turbo_encode_transmitted_symbols(information_bits)

            transmitted_systematic_symbols = 1.0 - 2.0 * encoded["systematic_stream_1"]
            transmitted_parity_symbols_1 = 1.0 - 2.0 * encoded["transmitted_parity_stream_1"]
            transmitted_parity_symbols_2 = 1.0 - 2.0 * encoded["transmitted_parity_stream_2"]

            received_systematic_symbols = transmitted_systematic_symbols + noise_standard_deviation * random_generator.standard_normal(total_encoded_length)
            received_parity_symbols_1 = transmitted_parity_symbols_1 + noise_standard_deviation * random_generator.standard_normal(len(transmitted_parity_symbols_1))
            received_parity_symbols_2 = transmitted_parity_symbols_2 + noise_standard_deviation * random_generator.standard_normal(len(transmitted_parity_symbols_2))

            received_parity_stream_1_full = depuncture_received_parity(received_parity_symbols_1, encoded["parity_keep_mask_1"])
            received_parity_stream_2_full = depuncture_received_parity(received_parity_symbols_2, encoded["parity_keep_mask_2"])

            _, llr_history = decode_turbo(
                received_systematic_stream_1=received_systematic_symbols,
                received_parity_stream_1_full=received_parity_stream_1_full,
                received_parity_stream_2_full=received_parity_stream_2_full,
                noise_variance=noise_variance,
                iteration_count=max(DECODER_ITERATION_LIST),
            )

            for iteration_count in DECODER_ITERATION_LIST:
                hard_information_bits = (llr_history[iteration_count - 1] < 0.0).astype(np.int8)
                bit_errors[iteration_count] += int(np.sum(information_bits != hard_information_bits))

            if ebn0_db == TURBO_EB_NO_DB[-1] and frame_count == 0:
                for iteration_count in DECODER_ITERATION_LIST:
                    llr_snapshot[iteration_count] = llr_history[iteration_count - 1][:20].copy()

            transmitted_information_bits += INFORMATION_BLOCK_LENGTH
            frame_count += 1

        for iteration_count in DECODER_ITERATION_LIST:
            ber_by_iteration[iteration_count].append(
                bit_errors[iteration_count] / transmitted_information_bits
            )

        print(
            "turbo rate={} Eb/N0={:4.2f} dB frames={} ".format(SELECTED_CODE_RATE, ebn0_db, frame_count)
            + ", ".join(
                [
                    f"it{iteration_count}={ber_by_iteration[iteration_count][-1]:.4e}"
                    for iteration_count in DECODER_ITERATION_LIST
                ]
            )
        )

    for iteration_count in DECODER_ITERATION_LIST:
        ber_by_iteration[iteration_count] = np.array(ber_by_iteration[iteration_count], dtype=float)

    return ber_by_iteration, llr_snapshot

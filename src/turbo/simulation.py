"""Simulation and plotting functions for the turbo-code project.

This module performs the Monte Carlo BER measurements and generates the figures used for the
comparison study.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

from config import (
    CONVOLUTIONAL_EB_NO_DB,
    CONVOLUTIONAL_MAXIMUM_FRAMES,
    CONVOLUTIONAL_MINIMUM_FRAMES,
    CONVOLUTIONAL_TARGET_ERRORS,
    DECODER_ITERATION_LIST,
    FIGURE_FILE_PREFIX,
    INFORMATION_BLOCK_LENGTH,
    NUMBER_OF_TAIL_BITS,
    SAVE_FIGURES,
    TURBO_EB_NO_DB,
    TURBO_MAXIMUM_FRAMES,
    TURBO_MINIMUM_FRAMES,
    TURBO_TARGET_ERRORS,
)
from decoder import decode_convolutional_75_viterbi, turbo_decode
from encoder import encode_convolutional_75, turbo_encode


# --------------------------------------------------------------------------------------
# Noise-variance helper
# --------------------------------------------------------------------------------------
# For BPSK over AWGN with code rate R and Eb/N0 in linear units,
# sigma^2 = 1 / (2 * R * Eb/N0).

def compute_noise_variance_from_eb_no(eb_no_db, code_rate):
    eb_no_linear = 10.0 ** (eb_no_db / 10.0)
    return 1.0 / (2.0 * code_rate * eb_no_linear)


# --------------------------------------------------------------------------------------
# Convolutional baseline simulation
# --------------------------------------------------------------------------------------
# This generates a reference curve so the turbo-code results can later be compared against
# a simpler coding method.

def run_convolutional_simulation(random_number_generator):
    bit_error_rate_values = []

    for eb_no_db in CONVOLUTIONAL_EB_NO_DB:
        noise_variance = compute_noise_variance_from_eb_no(eb_no_db, 0.5)
        noise_standard_deviation = np.sqrt(noise_variance)

        total_bit_errors = 0
        total_information_bits = 0
        frame_counter = 0

        while (
            frame_counter < CONVOLUTIONAL_MINIMUM_FRAMES
            or (total_bit_errors < CONVOLUTIONAL_TARGET_ERRORS and frame_counter < CONVOLUTIONAL_MAXIMUM_FRAMES)
        ):
            information_bits = random_number_generator.integers(
                0, 2, INFORMATION_BLOCK_LENGTH, dtype=np.int8
            )
            coded_bits = encode_convolutional_75(information_bits)
            transmitted_symbols = 1.0 - 2.0 * coded_bits
            received_symbols = transmitted_symbols + noise_standard_deviation * random_number_generator.standard_normal(len(transmitted_symbols))

            decoded_bits = decode_convolutional_75_viterbi(received_symbols, INFORMATION_BLOCK_LENGTH)
            total_bit_errors += int(np.sum(information_bits != decoded_bits))
            total_information_bits += INFORMATION_BLOCK_LENGTH
            frame_counter += 1

        bit_error_rate = total_bit_errors / total_information_bits
        bit_error_rate_values.append(bit_error_rate)
        print(
            f"convolutional Eb/N0={eb_no_db:4.1f} dB "
            f"BER={bit_error_rate:.4e} frames={frame_counter}"
        )

    return np.array(bit_error_rate_values, dtype=float)


# --------------------------------------------------------------------------------------
# Turbo-code simulation
# --------------------------------------------------------------------------------------
# A single random interleaver is fixed for the whole BER experiment. For each SNR point,
# the decoder is run for multiple iteration counts so the waterfall improvement can be seen.

def run_turbo_simulation(random_number_generator):
    interleaver_pattern = random_number_generator.permutation(INFORMATION_BLOCK_LENGTH)
    bit_error_rate_by_iteration = {iteration: [] for iteration in DECODER_ITERATION_LIST}
    llr_snapshot_by_iteration = {iteration: None for iteration in DECODER_ITERATION_LIST}

    total_stream_length = INFORMATION_BLOCK_LENGTH + NUMBER_OF_TAIL_BITS
    effective_code_rate = INFORMATION_BLOCK_LENGTH / (3 * INFORMATION_BLOCK_LENGTH + 4)

    for eb_no_db in TURBO_EB_NO_DB:
        noise_variance = compute_noise_variance_from_eb_no(eb_no_db, effective_code_rate)
        noise_standard_deviation = np.sqrt(noise_variance)

        error_count_by_iteration = {iteration: 0 for iteration in DECODER_ITERATION_LIST}
        total_information_bits = 0
        frame_counter = 0

        while (
            frame_counter < TURBO_MINIMUM_FRAMES
            or (
                error_count_by_iteration[max(DECODER_ITERATION_LIST)] < TURBO_TARGET_ERRORS
                and frame_counter < TURBO_MAXIMUM_FRAMES
            )
        ):
            information_bits = random_number_generator.integers(
                0, 2, INFORMATION_BLOCK_LENGTH, dtype=np.int8
            )
            systematic_stream_one, parity_stream_one, _, parity_stream_two = turbo_encode(
                information_bits,
                interleaver_pattern,
            )

            transmitted_systematic_stream = 1.0 - 2.0 * systematic_stream_one
            transmitted_parity_stream_one = 1.0 - 2.0 * parity_stream_one
            transmitted_parity_stream_two = 1.0 - 2.0 * parity_stream_two

            received_systematic_stream = transmitted_systematic_stream + noise_standard_deviation * random_number_generator.standard_normal(total_stream_length)
            received_parity_stream_one = transmitted_parity_stream_one + noise_standard_deviation * random_number_generator.standard_normal(total_stream_length)
            received_parity_stream_two = transmitted_parity_stream_two + noise_standard_deviation * random_number_generator.standard_normal(total_stream_length)

            _, iteration_llr_history = turbo_decode(
                received_systematic_stream,
                received_parity_stream_one,
                received_parity_stream_two,
                interleaver_pattern,
                noise_variance,
                max(DECODER_ITERATION_LIST),
            )

            for iteration in DECODER_ITERATION_LIST:
                hard_decision_bits = (iteration_llr_history[iteration - 1] < 0.0).astype(np.int8)
                error_count_by_iteration[iteration] += int(np.sum(information_bits != hard_decision_bits))

            if eb_no_db == TURBO_EB_NO_DB[-1] and frame_counter == 0:
                for iteration in DECODER_ITERATION_LIST:
                    llr_snapshot_by_iteration[iteration] = iteration_llr_history[iteration - 1][:20].copy()

            total_information_bits += INFORMATION_BLOCK_LENGTH
            frame_counter += 1

        for iteration in DECODER_ITERATION_LIST:
            bit_error_rate_by_iteration[iteration].append(
                error_count_by_iteration[iteration] / total_information_bits
            )

        print(
            f"turbo Eb/N0={eb_no_db:4.1f} dB frames={frame_counter} "
            + ", ".join(
                [
                    f"iteration{iteration}={bit_error_rate_by_iteration[iteration][-1]:.4e}"
                    for iteration in DECODER_ITERATION_LIST
                ]
            )
        )

    for iteration in DECODER_ITERATION_LIST:
        bit_error_rate_by_iteration[iteration] = np.array(
            bit_error_rate_by_iteration[iteration], dtype=float
        )

    return bit_error_rate_by_iteration, llr_snapshot_by_iteration


# --------------------------------------------------------------------------------------
# Plot generation
# --------------------------------------------------------------------------------------
# The first plot compares uncoded BPSK with the convolutional baseline.
# The second plot shows how turbo-code BER changes with decoding iterations.
# The third plot shows the growth of posterior LLR confidence across iterations.

def plot_simulation_results(convolutional_bit_error_rate, turbo_bit_error_rate_by_iteration, llr_snapshot_by_iteration):
    plot_floor = 1e-8

    plt.figure(figsize=(7, 4.8))
    uncoded_bpsk_bit_error_rate = 0.5 * erfc(np.sqrt(10.0 ** (CONVOLUTIONAL_EB_NO_DB / 10.0)))
    plt.semilogy(
        CONVOLUTIONAL_EB_NO_DB,
        np.clip(uncoded_bpsk_bit_error_rate, plot_floor, None),
        'r.-',
        label='Uncoded BPSK',
    )
    plt.semilogy(
        CONVOLUTIONAL_EB_NO_DB,
        np.clip(convolutional_bit_error_rate, plot_floor, None),
        'bo-',
        label='Convolutional code (7,5)',
    )
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.title('Convolutional baseline')
    plt.grid(True, which='both', alpha=0.35)
    plt.legend()
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(f'{FIGURE_FILE_PREFIX}_convolutional.png', dpi=160)
    plt.show()

    plt.figure(figsize=(7, 4.8))
    for iteration in DECODER_ITERATION_LIST:
        plt.semilogy(
            TURBO_EB_NO_DB,
            np.clip(turbo_bit_error_rate_by_iteration[iteration], plot_floor, None),
            marker='x',
            label=f'Iteration {iteration}',
        )
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.title('Turbo code BER (terminated recursive systematic code, max-log-MAP)')
    plt.grid(True, which='both', alpha=0.35)
    plt.legend()
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(f'{FIGURE_FILE_PREFIX}_turbo.png', dpi=160)
    plt.show()

    plt.figure(figsize=(8, 5.2))
    for iteration in DECODER_ITERATION_LIST:
        if llr_snapshot_by_iteration[iteration] is not None:
            plt.scatter(
                range(len(llr_snapshot_by_iteration[iteration])),
                llr_snapshot_by_iteration[iteration],
                s=18,
                label=f'Iteration {iteration}',
            )
    plt.xlabel('Bit index')
    plt.ylabel('Posterior LLR')
    plt.title('Turbo decoder confidence growth')
    plt.grid(True, linestyle='--', alpha=0.45)
    plt.legend()
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(f'{FIGURE_FILE_PREFIX}_llr.png', dpi=160)
    plt.show()

import numpy as np

from config import (
    RANDOM_SEED,
    INFORMATION_BIT_COUNT,
    CODE_RATE,
    EBN0_DECIBELS,
    ITERATION_LIST,
    MINIMUM_FRAME_COUNT,
    MAXIMUM_FRAME_COUNT,
    TARGET_ERROR_COUNT,
)
from encoder import encode_information_bits, verify_codeword
from decoder import decode_codeword

# ============================================================
# Noise model
# ============================================================
# We use the same BPSK and AWGN conventions as in the turbo project:
#   0 -> +1
#   1 -> -1
# and
#   sigma^2 = 1 / (2 R Eb/N0)
# where R is the code rate.


def compute_noise_variance_from_ebn0(ebn0_decibels, code_rate):
    """Convert Eb/N0 in dB to AWGN noise variance."""
    ebn0_linear = 10.0 ** (ebn0_decibels / 10.0)
    return 1.0 / (2.0 * code_rate * ebn0_linear)



def map_bits_to_bpsk_symbols(bit_array):
    """Map binary bits to BPSK symbols using 0 -> +1 and 1 -> -1."""
    return 1.0 - 2.0 * np.asarray(bit_array, dtype=float)


# ============================================================
# Monte Carlo simulation
# ============================================================
# The simulation collects BER points for several iteration counts so that
# the LDPC decoder can be compared directly against the turbo decoder.


def run_ldpc_simulation(random_seed=RANDOM_SEED):
    """Run BER simulation for the LDPC code over AWGN.

    Returns
    -------
    bit_error_rate_results:
        Dictionary mapping iteration number to BER curve.
    llr_snapshot_by_iteration:
        Posterior LLR samples from one high-SNR frame for plotting.
    """
    random_generator = np.random.default_rng(random_seed)

    bit_error_rate_results = {iteration_count: [] for iteration_count in ITERATION_LIST}
    llr_snapshot_by_iteration = {iteration_count: None for iteration_count in ITERATION_LIST}

    for ebn0_decibels in EBN0_DECIBELS:
        noise_variance = compute_noise_variance_from_ebn0(ebn0_decibels, CODE_RATE)
        noise_standard_deviation = np.sqrt(noise_variance)

        bit_errors_by_iteration = {iteration_count: 0 for iteration_count in ITERATION_LIST}
        processed_information_bits = 0
        processed_frames = 0

        while (
            processed_frames < MINIMUM_FRAME_COUNT
            or (
                bit_errors_by_iteration[max(ITERATION_LIST)] < TARGET_ERROR_COUNT
                and processed_frames < MAXIMUM_FRAME_COUNT
            )
        ):
            information_bits = random_generator.integers(
                0, 2, INFORMATION_BIT_COUNT, dtype=np.int8
            )

            codeword_bits = encode_information_bits(information_bits)
            if not verify_codeword(codeword_bits):
                raise RuntimeError("Encoder produced a vector that does not satisfy Hc^T = 0.")

            transmitted_symbols = map_bits_to_bpsk_symbols(codeword_bits)
            received_samples = transmitted_symbols + noise_standard_deviation * random_generator.standard_normal(
                len(transmitted_symbols)
            )

            _, posterior_llr_history = decode_codeword(
                received_samples,
                noise_variance,
                max(ITERATION_LIST),
            )

            for iteration_count in ITERATION_LIST:
                hard_information_bits = (
                    posterior_llr_history[iteration_count - 1][:INFORMATION_BIT_COUNT] < 0.0
                ).astype(np.int8)
                bit_errors_by_iteration[iteration_count] += int(
                    np.sum(information_bits != hard_information_bits)
                )

            if ebn0_decibels == EBN0_DECIBELS[-1] and processed_frames == 0:
                for iteration_count in ITERATION_LIST:
                    llr_snapshot_by_iteration[iteration_count] = posterior_llr_history[
                        iteration_count - 1
                    ][:20].copy()

            processed_information_bits += INFORMATION_BIT_COUNT
            processed_frames += 1

        for iteration_count in ITERATION_LIST:
            bit_error_rate_results[iteration_count].append(
                bit_errors_by_iteration[iteration_count] / processed_information_bits
            )

        print(
            "LDPC Eb/N0={:4.2f} dB frames={} ".format(ebn0_decibels, processed_frames)
            + ", ".join(
                [
                    f"iteration {iteration_count}={bit_error_rate_results[iteration_count][-1]:.4e}"
                    for iteration_count in ITERATION_LIST
                ]
            )
        )

    for iteration_count in ITERATION_LIST:
        bit_error_rate_results[iteration_count] = np.array(
            bit_error_rate_results[iteration_count], dtype=float
        )

    return bit_error_rate_results, llr_snapshot_by_iteration

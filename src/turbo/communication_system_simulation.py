import numpy as np
from configuration_parameters import *
from convolutional_and_turbo_encoders import *
from viterbi_and_turbo_decoders import *

# ============================================================
# NOISE VARIANCE CALCULATION
# ============================================================

def compute_noise_variance_from_ebn0(ebn0_db, code_rate):
    """
    Convert Eb/N0 in dB to noise variance sigma^2 for AWGN channel
    Args:
        ebn0_db: Eb/N0 in dB
        code_rate: code rate (k/n)
    Returns:
        noise variance
    """
    ebn0_linear = 10**(ebn0_db/10)
    return 1.0 / (2.0 * code_rate * ebn0_linear)


# ============================================================
# CONVOLUTIONAL CODE SIMULATION
# ============================================================

def run_convolutional_code_simulation(random_generator):
    """
    Run the simulation for convolutional code over AWGN channel.
    Returns:
        bit_error_rates: array of BER per Eb/N0
    """

    bit_error_rates = []

    for ebn0 in CONVOLUTIONAL_EBN0_VALUES_DB:

        noise_variance = compute_noise_variance_from_ebn0(ebn0, 0.5)
        noise_sigma = np.sqrt(noise_variance)

        bit_errors = 0
        total_bits = 0
        frames = 0

        while frames < CONVOLUTIONAL_MINIMUM_FRAMES:

            # Generate random information bits
            information_bits = random_generator.integers(0, 2, INFORMATION_BLOCK_LENGTH, dtype=np.int8)

            # Encode
            encoded_bits = convolutional_encode_generator_7_5(information_bits)

            # Map to BPSK
            transmitted_symbols = 1.0 - 2.0 * encoded_bits

            # Pass through AWGN
            received_symbols = transmitted_symbols + noise_sigma * random_generator.standard_normal(len(transmitted_symbols))

            # Decode
            decoded_bits = viterbi_decode_convolutional_code(received_symbols, INFORMATION_BLOCK_LENGTH)

            # Count bit errors
            bit_errors += np.sum(information_bits != decoded_bits)
            total_bits += INFORMATION_BLOCK_LENGTH
            frames += 1

        ber = bit_errors / total_bits
        bit_error_rates.append(ber)

        print(f"Eb/N0={ebn0} dB, BER={ber:.4e}")

    return np.array(bit_error_rates)

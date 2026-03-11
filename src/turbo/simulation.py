import numpy as np
from encoder import turbo_encode
from decoder import turbo_decode
from scipy.special import erfc

# -----------------------------
# Utility functions for AWGN simulation
# -----------------------------
def noise_variance_from_ebn0(ebn0_db, code_rate):
    """
    Convert Eb/N0 (dB) to noise variance for BPSK simulation
    """
    ebn0_linear = 10.0 ** (ebn0_db / 10.0)
    return 1.0 / (2.0 * code_rate * ebn0_linear)


def monte_carlo_turbo(rng, information_length, interleaver, ebn0_db_array, iterations_list, min_frames=80, max_frames=500, target_errors=150):
    """
    Monte Carlo simulation for Turbo code
    """
    turbo_results = {iteration: [] for iteration in iterations_list}
    llr_snapshot = {iteration: None for iteration in iterations_list}

    total_length = information_length + 2  # including tail bits
    effective_code_rate = information_length / (3 * information_length + 4)

    for ebn0_db in ebn0_db_array:
        noise_variance = noise_variance_from_ebn0(ebn0_db, effective_code_rate)
        sigma = np.sqrt(noise_variance)
        errors = {iteration: 0 for iteration in iterations_list}
        total_bits = 0
        frames = 0

        while frames < min_frames or (errors[max(iterations_list)] < target_errors and frames < max_frames):
            information_bits = rng.integers(0, 2, information_length, dtype=np.int8)
            systematic_1, parity_1, systematic_2, parity_2 = turbo_encode(information_bits, interleaver)

            tx_systematic = 1.0 - 2.0 * systematic_1
            tx_parity1 = 1.0 - 2.0 * parity_1
            tx_parity2 = 1.0 - 2.0 * parity_2

            rx_systematic = tx_systematic + sigma * rng.standard_normal(total_length)
            rx_parity1 = tx_parity1 + sigma * rng.standard_normal(total_length)
            rx_parity2 = tx_parity2 + sigma * rng.standard_normal(total_length)

            hard_bits, llr_iterations = turbo_decode(rx_systematic, rx_parity1, rx_parity2, interleaver, noise_variance, max(iterations_list))

            for iteration in iterations_list:
                hard_decision = (llr_iterations[iteration - 1] < 0.0).astype(np.int8)
                errors[iteration] += np.sum(information_bits != hard_decision)

            if ebn0_db == ebn0_db_array[-1] and frames == 0:
                for iteration in iterations_list:
                    llr_snapshot[iteration] = llr_iterations[iteration - 1][:20].copy()

            total_bits += information_length
            frames += 1

        for iteration in iterations_list:
            turbo_results[iteration].append(errors[iteration] / total_bits)
        print(f'Turbo Eb/N0={ebn0_db:.2f} dB, frames={frames}, ' +
              ', '.join([f'iter{it}={turbo_results[it][-1]:.4e}' for it in iterations_list]))

    for iteration in iterations_list:
        turbo_results[iteration] = np.array(turbo_results[iteration], dtype=float)

    return turbo_results, llr_snapshot

"""
LDPC decoder.

This file uses layered normalized min-sum decoding, which is a good practical
compromise between complexity and performance for sparse parity-check graphs.
"""

import numpy as np

from config import NORMALIZATION_FACTOR
from encoder import CHECK_TO_VARIABLE_NEIGHBORS, PARITY_CHECK_MATRIX


def channel_llr_from_received_symbols(received_symbols, noise_variance):
    """
    Compute AWGN channel LLR values for BPSK.
    """
    return (2.0 / noise_variance) * np.asarray(received_symbols, dtype=float)


def decode_codeword_with_layered_min_sum(
    received_symbols,
    noise_variance,
    iteration_count,
):
    """
    Layered normalized min-sum decoding.

    Returns:
    - hard_decision_bits
    - posterior_llr_history
    """
    channel_llr = channel_llr_from_received_symbols(received_symbols, noise_variance)
    posterior_llr = channel_llr.copy()
    posterior_llr_history = []

    check_to_variable_messages = [
        np.zeros(len(variable_indices), dtype=float)
        for variable_indices in CHECK_TO_VARIABLE_NEIGHBORS
    ]

    for _ in range(iteration_count):
        for check_index, variable_indices in enumerate(CHECK_TO_VARIABLE_NEIGHBORS):
            old_messages = check_to_variable_messages[check_index]
            extrinsic_values = posterior_llr[variable_indices] - old_messages

            signs = np.sign(extrinsic_values)
            signs[signs == 0.0] = 1.0
            absolute_values = np.abs(extrinsic_values)

            smallest_index = int(np.argmin(absolute_values))
            smallest_value = absolute_values[smallest_index]

            masked_values = absolute_values.copy()
            masked_values[smallest_index] = np.inf
            second_smallest_value = np.min(masked_values)

            total_sign = np.prod(signs)
            new_messages = np.zeros_like(old_messages)

            for local_index in range(len(variable_indices)):
                magnitude = second_smallest_value if local_index == smallest_index else smallest_value
                new_messages[local_index] = (
                    NORMALIZATION_FACTOR
                    * total_sign
                    * signs[local_index]
                    * magnitude
                )

            check_to_variable_messages[check_index] = new_messages
            posterior_llr[variable_indices] = extrinsic_values + new_messages

        posterior_llr_history.append(posterior_llr.copy())

    hard_decision_bits = (posterior_llr < 0.0).astype(np.int8)
    return hard_decision_bits, posterior_llr_history

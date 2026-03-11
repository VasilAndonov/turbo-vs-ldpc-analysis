import numpy as np

from config import (
    CODEWORD_BIT_COUNT,
    INFORMATION_BIT_COUNT,
    NORMALIZATION_FACTOR,
)
from encoder import (
    PARITY_CHECK_MATRIX,
    CHECK_TO_VARIABLE_NEIGHBORS,
    VARIABLE_TO_CHECK_NEIGHBORS,
)

# ============================================================
# Channel LLR conversion
# ============================================================
# For BPSK over AWGN with mapping 0 -> +1 and 1 -> -1, the channel LLR is:
#
#     L = 2 y / sigma^2
#
# where y is the received symbol and sigma^2 is the noise variance.


def convert_received_samples_to_llr(received_samples, noise_variance):
    """Convert noisy BPSK samples into channel log-likelihood ratios."""
    received_samples = np.asarray(received_samples, dtype=float)
    return (2.0 / noise_variance) * received_samples


# ============================================================
# Check-node update
# ============================================================
# We use normalized min-sum decoding. For each outgoing check-to-variable
# message, we combine the signs of all incoming messages and take the
# smallest magnitude among the others. The normalization factor slightly
# compensates for the min-sum approximation.


def compute_check_to_variable_messages(variable_to_check_messages):
    """Perform one check-node update for all edges."""
    check_to_variable_messages = {}

    for check_index, variable_neighbors in enumerate(CHECK_TO_VARIABLE_NEIGHBORS):
        incoming_messages = np.array(
            [variable_to_check_messages[(check_index, variable_index)] for variable_index in variable_neighbors],
            dtype=float,
        )

        incoming_signs = np.sign(incoming_messages)
        incoming_signs[incoming_signs == 0.0] = 1.0
        incoming_magnitudes = np.abs(incoming_messages)

        total_sign = np.prod(incoming_signs)

        for local_position, variable_index in enumerate(variable_neighbors):
            other_sign = total_sign * incoming_signs[local_position]

            if len(variable_neighbors) == 1:
                minimum_other_magnitude = 0.0
            else:
                mask = np.ones(len(variable_neighbors), dtype=bool)
                mask[local_position] = False
                minimum_other_magnitude = np.min(incoming_magnitudes[mask])

            check_to_variable_messages[(check_index, variable_index)] = (
                NORMALIZATION_FACTOR * other_sign * minimum_other_magnitude
            )

    return check_to_variable_messages


# ============================================================
# Variable-node update and posterior LLR computation
# ============================================================
# Each variable node combines:
#   - the original channel LLR,
#   - all incoming check-node messages.
#
# The outgoing message to one check node excludes the message from that
# same check node.


def compute_variable_to_check_messages(channel_llr_values, check_to_variable_messages):
    """Perform one variable-node update for all edges."""
    variable_to_check_messages = {}

    for variable_index, check_neighbors in enumerate(VARIABLE_TO_CHECK_NEIGHBORS):
        total_message = channel_llr_values[variable_index]
        for check_index in check_neighbors:
            total_message += check_to_variable_messages[(check_index, variable_index)]

        for check_index in check_neighbors:
            variable_to_check_messages[(check_index, variable_index)] = (
                total_message - check_to_variable_messages[(check_index, variable_index)]
            )

    return variable_to_check_messages



def compute_posterior_llr(channel_llr_values, check_to_variable_messages):
    """Combine all information into posterior LLR values."""
    posterior_llr_values = channel_llr_values.astype(float).copy()

    for variable_index, check_neighbors in enumerate(VARIABLE_TO_CHECK_NEIGHBORS):
        for check_index in check_neighbors:
            posterior_llr_values[variable_index] += check_to_variable_messages[(check_index, variable_index)]

    return posterior_llr_values



def compute_syndrome_from_hard_decision(hard_decision_bits):
    """Check whether the current hard decision satisfies all parity checks."""
    return (PARITY_CHECK_MATRIX @ hard_decision_bits) % 2


# ============================================================
# Full LDPC belief-propagation decoder
# ============================================================
# The decoder stores the posterior LLR after every iteration so that the
# simulation can generate curves for 1, 2, ..., N iterations, just like
# the turbo-code project.


def decode_codeword(received_samples, noise_variance, number_of_iterations):
    """Decode a noisy LDPC codeword with normalized min-sum BP decoding.

    Returns
    -------
    information_bits_hard_decision:
        Hard decision on the information part after the final iteration.
    posterior_llr_history:
        List containing the full codeword posterior LLR after each iteration.
    """
    received_samples = np.asarray(received_samples, dtype=float)
    if len(received_samples) != CODEWORD_BIT_COUNT:
        raise ValueError(
            f"Expected {CODEWORD_BIT_COUNT} received samples, got {len(received_samples)}."
        )

    channel_llr_values = convert_received_samples_to_llr(received_samples, noise_variance)

    variable_to_check_messages = {}
    for check_index, variable_neighbors in enumerate(CHECK_TO_VARIABLE_NEIGHBORS):
        for variable_index in variable_neighbors:
            variable_to_check_messages[(check_index, variable_index)] = channel_llr_values[variable_index]

    posterior_llr_history = []

    for _ in range(number_of_iterations):
        check_to_variable_messages = compute_check_to_variable_messages(variable_to_check_messages)
        posterior_llr_values = compute_posterior_llr(channel_llr_values, check_to_variable_messages)
        posterior_llr_history.append(posterior_llr_values.copy())

        hard_decision_bits = (posterior_llr_values < 0.0).astype(np.int8)
        syndrome = compute_syndrome_from_hard_decision(hard_decision_bits)

        if np.all(syndrome == 0):
            # Fill the remaining history with the converged result so that
            # later plotting code can still index every iteration.
            while len(posterior_llr_history) < number_of_iterations:
                posterior_llr_history.append(posterior_llr_values.copy())
            break

        variable_to_check_messages = compute_variable_to_check_messages(
            channel_llr_values, check_to_variable_messages
        )

    final_hard_decision = (posterior_llr_history[-1][:INFORMATION_BIT_COUNT] < 0.0).astype(np.int8)
    return final_hard_decision, posterior_llr_history

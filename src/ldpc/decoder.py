"""
LDPC decoder.
"""

import numpy as np
from config import LLR_CLIP, MESSAGE_DAMPING
from encoder import CHECK_TO_VARIABLE_NEIGHBORS, PARITY_CHECK_MATRIX, VARIABLE_TO_CHECK_NEIGHBORS


def channel_llr_from_received_symbols(received_symbols, noise_variance):
    return np.clip((2.0 / noise_variance) * np.asarray(received_symbols, dtype=float), -LLR_CLIP, LLR_CLIP)


def compute_syndrome(hard_bits):
    return (PARITY_CHECK_MATRIX @ hard_bits) % 2


def decode_codeword_with_sum_product(received_symbols, noise_variance, iteration_count):
    channel_llr = channel_llr_from_received_symbols(received_symbols, noise_variance)
    variable_count = len(channel_llr)

    v_to_c = {}
    c_to_v = {}
    for v in range(variable_count):
        for c in VARIABLE_TO_CHECK_NEIGHBORS[v]:
            v_to_c[(c, v)] = channel_llr[v]
            c_to_v[(c, v)] = 0.0

    posterior_history = []
    posterior = channel_llr.copy()

    for _ in range(iteration_count):
        new_c_to_v = {}
        for c, var_indices in enumerate(CHECK_TO_VARIABLE_NEIGHBORS):
            incoming = np.array([v_to_c[(c, v)] for v in var_indices], dtype=float)
            tanh_vals = np.tanh(np.clip(incoming / 2.0, -10.0, 10.0))

            for i, v in enumerate(var_indices):
                if len(var_indices) == 1:
                    updated = 0.0
                else:
                    others = np.delete(tanh_vals, i)
                    product = np.prod(others)
                    product = np.clip(product, -0.999999, 0.999999)
                    updated = 2.0 * np.arctanh(product)

                prev = c_to_v[(c, v)]
                damped = (1.0 - MESSAGE_DAMPING) * updated + MESSAGE_DAMPING * prev
                new_c_to_v[(c, v)] = np.clip(damped, -LLR_CLIP, LLR_CLIP)

        c_to_v = new_c_to_v

        posterior = channel_llr.copy()
        for v in range(variable_count):
            for c in VARIABLE_TO_CHECK_NEIGHBORS[v]:
                posterior[v] += c_to_v[(c, v)]
        posterior = np.clip(posterior, -LLR_CLIP, LLR_CLIP)
        posterior_history.append(posterior.copy())

        for v in range(variable_count):
            for c in VARIABLE_TO_CHECK_NEIGHBORS[v]:
                ext = posterior[v] - c_to_v[(c, v)]
                v_to_c[(c, v)] = np.clip(ext, -LLR_CLIP, LLR_CLIP)

        hard = (posterior < 0.0).astype(np.int8)
        if np.all(compute_syndrome(hard) == 0):
            while len(posterior_history) < iteration_count:
                posterior_history.append(posterior.copy())
            return hard, posterior_history

    hard = (posterior < 0.0).astype(np.int8)
    return hard, posterior_history

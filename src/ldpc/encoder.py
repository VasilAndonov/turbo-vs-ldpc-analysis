"""
Encoder-side LDPC utilities.

This file contains three main parts:
1) construction of a better parity-check matrix,
2) graph utilities used by the decoder,
3) systematic LDPC encoding.
"""

import numpy as np

from config import (
    INFORMATION_BIT_COUNT,
    INFORMATION_COLUMN_WEIGHT,
    PARITY_BIT_COUNT,
    RANDOM_SEED,
    ROW_WEIGHT_PENALTY,
)


def build_information_connection_matrix():
    """Create the sparse left side A of the parity-check matrix."""
    random_generator = np.random.default_rng(RANDOM_SEED)

    information_connection_matrix = np.zeros(
        (PARITY_BIT_COUNT, INFORMATION_BIT_COUNT), dtype=np.int8
    )

    row_weights = np.zeros(PARITY_BIT_COUNT, dtype=int)
    used_row_pairs = np.zeros((PARITY_BIT_COUNT, PARITY_BIT_COUNT), dtype=np.int16)

    for column_index in range(INFORMATION_BIT_COUNT):
        chosen_rows = []

        for _ in range(INFORMATION_COLUMN_WEIGHT):
            best_score = None
            best_row_candidates = []

            for row_index in range(PARITY_BIT_COUNT):
                if row_index in chosen_rows:
                    continue

                pair_penalty = 0
                for selected_row in chosen_rows:
                    low = min(row_index, selected_row)
                    high = max(row_index, selected_row)
                    pair_penalty += used_row_pairs[low, high]

                score = pair_penalty + ROW_WEIGHT_PENALTY * row_weights[row_index]

                if best_score is None or score < best_score:
                    best_score = score
                    best_row_candidates = [row_index]
                elif score == best_score:
                    best_row_candidates.append(row_index)

            chosen_row = random_generator.choice(best_row_candidates)
            chosen_rows.append(int(chosen_row))

        chosen_rows.sort()

        for row_index in chosen_rows:
            information_connection_matrix[row_index, column_index] = 1
            row_weights[row_index] += 1

        for first_index in range(len(chosen_rows)):
            for second_index in range(first_index + 1, len(chosen_rows)):
                row_a = chosen_rows[first_index]
                row_b = chosen_rows[second_index]
                used_row_pairs[row_a, row_b] += 1

    return information_connection_matrix


def build_parity_connection_matrix():
    """Create the sparse right side B of the parity-check matrix."""
    parity_connection_matrix = np.zeros(
        (PARITY_BIT_COUNT, PARITY_BIT_COUNT), dtype=np.int8
    )

    for row_index in range(PARITY_BIT_COUNT):
        parity_connection_matrix[row_index, row_index] = 1

        if row_index >= 1:
            parity_connection_matrix[row_index, row_index - 1] = 1

        if row_index >= 2 and row_index % 3 != 0:
            parity_connection_matrix[row_index, row_index - 2] = 1

    return parity_connection_matrix


def build_parity_check_matrix():
    information_connection_matrix = build_information_connection_matrix()
    parity_connection_matrix = build_parity_connection_matrix()

    parity_check_matrix = np.concatenate(
        [information_connection_matrix, parity_connection_matrix], axis=1
    )
    return parity_check_matrix, information_connection_matrix, parity_connection_matrix


PARITY_CHECK_MATRIX, INFORMATION_CONNECTION_MATRIX, PARITY_CONNECTION_MATRIX = (
    build_parity_check_matrix()
)


def build_graph_from_parity_check_matrix(parity_check_matrix):
    """Convert H into row-to-column and column-to-row adjacency lists."""
    check_to_variable_neighbors = []
    variable_to_check_neighbors = [[] for _ in range(parity_check_matrix.shape[1])]

    for check_index in range(parity_check_matrix.shape[0]):
        variable_indices = list(np.where(parity_check_matrix[check_index] == 1)[0])
        check_to_variable_neighbors.append(variable_indices)

        for variable_index in variable_indices:
            variable_to_check_neighbors[variable_index].append(check_index)

    return check_to_variable_neighbors, variable_to_check_neighbors


CHECK_TO_VARIABLE_NEIGHBORS, VARIABLE_TO_CHECK_NEIGHBORS = (
    build_graph_from_parity_check_matrix(PARITY_CHECK_MATRIX)
)


def solve_lower_triangular_binary_system(lower_triangular_matrix, right_hand_side):
    """Solve B p = s over GF(2) by forward substitution."""
    size = len(right_hand_side)
    solution = np.zeros(size, dtype=np.int8)

    for row_index in range(size):
        value = int(right_hand_side[row_index])

        for column_index in range(row_index):
            if lower_triangular_matrix[row_index, column_index]:
                value ^= int(solution[column_index])

        solution[row_index] = value

    return solution


def encode_information_bits(information_bits):
    """Encode one information block into a systematic LDPC codeword."""
    information_bits = np.asarray(information_bits, dtype=np.int8)

    syndrome_from_information = (
        INFORMATION_CONNECTION_MATRIX @ information_bits
    ) % 2

    parity_bits = solve_lower_triangular_binary_system(
        PARITY_CONNECTION_MATRIX, syndrome_from_information
    )

    return np.concatenate([information_bits, parity_bits]).astype(np.int8)

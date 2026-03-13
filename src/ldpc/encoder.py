"""
LDPC encoder and graph construction.
"""

import numpy as np
from config import (
    INFORMATION_BIT_COUNT,
    PARITY_BIT_COUNT,
    RANDOM_SEED,
    ROW_WEIGHT_PENALTY,
    get_information_column_weight,
    get_parity_band_width,
)


def build_information_connection_matrix():
    rng = np.random.default_rng(RANDOM_SEED)
    weight = get_information_column_weight()
    if weight > PARITY_BIT_COUNT:
        raise ValueError(f"Column weight {weight} is too large for {PARITY_BIT_COUNT} parity checks.")

    A = np.zeros((PARITY_BIT_COUNT, INFORMATION_BIT_COUNT), dtype=np.int8)
    row_weights = np.zeros(PARITY_BIT_COUNT, dtype=int)
    used_pairs = np.zeros((PARITY_BIT_COUNT, PARITY_BIT_COUNT), dtype=np.int16)

    for col in range(INFORMATION_BIT_COUNT):
        chosen = []
        for _ in range(weight):
            best_score = None
            best = []
            for row in range(PARITY_BIT_COUNT):
                if row in chosen:
                    continue
                pair_penalty = 0.0
                for s in chosen:
                    lo, hi = min(row, s), max(row, s)
                    pair_penalty += used_pairs[lo, hi]
                score = pair_penalty + ROW_WEIGHT_PENALTY * row_weights[row]
                if best_score is None or score < best_score:
                    best_score = score
                    best = [row]
                elif score == best_score:
                    best.append(row)
            chosen.append(int(rng.choice(best)))

        chosen.sort()
        for row in chosen:
            A[row, col] = 1
            row_weights[row] += 1
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                used_pairs[chosen[i], chosen[j]] += 1

    return A


def build_parity_connection_matrix():
    band = get_parity_band_width()
    B = np.zeros((PARITY_BIT_COUNT, PARITY_BIT_COUNT), dtype=np.int8)

    for row in range(PARITY_BIT_COUNT):
        B[row, row] = 1
        for offset in range(1, band + 1):
            if row - offset >= 0 and (((row + offset) % 2 == 0) or offset == 1):
                B[row, row - offset] = 1

    return B


def build_parity_check_matrix():
    A = build_information_connection_matrix()
    B = build_parity_connection_matrix()
    H = np.concatenate([A, B], axis=1)
    return H, A, B


PARITY_CHECK_MATRIX, INFORMATION_CONNECTION_MATRIX, PARITY_CONNECTION_MATRIX = build_parity_check_matrix()


def build_graph_from_parity_check_matrix(H):
    check_to_variable_neighbors = []
    variable_to_check_neighbors = [[] for _ in range(H.shape[1])]

    for chk in range(H.shape[0]):
        vars_ = list(np.where(H[chk] == 1)[0])
        check_to_variable_neighbors.append(vars_)
        for v in vars_:
            variable_to_check_neighbors[v].append(chk)

    return check_to_variable_neighbors, variable_to_check_neighbors


CHECK_TO_VARIABLE_NEIGHBORS, VARIABLE_TO_CHECK_NEIGHBORS = build_graph_from_parity_check_matrix(PARITY_CHECK_MATRIX)


def solve_lower_triangular_binary_system(B, rhs):
    n = len(rhs)
    sol = np.zeros(n, dtype=np.int8)
    for row in range(n):
        value = int(rhs[row])
        for col in range(row):
            if B[row, col]:
                value ^= int(sol[col])
        sol[row] = value
    return sol


def encode_information_bits(information_bits):
    information_bits = np.asarray(information_bits, dtype=np.int8)
    syndrome = (INFORMATION_CONNECTION_MATRIX @ information_bits) % 2
    parity_bits = solve_lower_triangular_binary_system(PARITY_CONNECTION_MATRIX, syndrome)
    return np.concatenate([information_bits, parity_bits]).astype(np.int8)

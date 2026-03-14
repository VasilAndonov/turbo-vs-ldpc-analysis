import numpy as np
from ldpc.config import INFORMATION_BITS, SUPPORTED_CODE_RATES, RANDOM_SEED

def build_ldpc_parameters(rate_label):
    if rate_label == "1/3":
        return dict(column_weight=7, band_width=6)
    if rate_label == "1/2":
        return dict(column_weight=6, band_width=5)
    if rate_label == "3/4":
        return dict(column_weight=5, band_width=4)
    if rate_label == "7/8":
        return dict(column_weight=4, band_width=3)
    raise ValueError(rate_label)

def build_ldpc_matrices(rate_label):
    params = build_ldpc_parameters(rate_label)
    column_weight = params["column_weight"]
    band_width = params["band_width"]
    code_rate = SUPPORTED_CODE_RATES[rate_label]
    parity_bits = int(round(INFORMATION_BITS * (1.0 / code_rate - 1.0)))
    codeword_bits = INFORMATION_BITS + parity_bits
    local_rng = np.random.default_rng(RANDOM_SEED + (abs(hash(rate_label)) % 1000))
    A = np.zeros((parity_bits, INFORMATION_BITS), dtype=np.int8)
    row_weights = np.zeros(parity_bits, dtype=np.int32)
    used_pairs = np.zeros((parity_bits, parity_bits), dtype=np.int16)
    base_stride = max(1, parity_bits // column_weight)

    for col in range(INFORMATION_BITS):
        target_rows = []
        start_row = (col * base_stride) % parity_bits
        for local_index in range(column_weight):
            target_rows.append((start_row + local_index * base_stride) % parity_bits)
        chosen = []
        for candidate_row in target_rows:
            window = [(candidate_row + offset) % parity_bits for offset in range(-2, 3)]
            best_score = None
            best_rows = []
            for row in window:
                if row in chosen:
                    continue
                pair_penalty = 0.0
                for s in chosen:
                    lo = min(row, s)
                    hi = max(row, s)
                    pair_penalty += used_pairs[lo, hi]
                score = pair_penalty + 0.55 * row_weights[row]
                if best_score is None or score < best_score:
                    best_score = score
                    best_rows = [row]
                elif score == best_score:
                    best_rows.append(row)
            chosen.append(int(local_rng.choice(best_rows)))
        chosen = sorted(set(chosen))
        while len(chosen) < column_weight:
            remaining = [r for r in range(parity_bits) if r not in chosen]
            chosen.append(int(local_rng.choice(remaining)))
            chosen = sorted(set(chosen))
        for row in chosen:
            A[row, col] = 1
            row_weights[row] += 1
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                used_pairs[chosen[i], chosen[j]] += 1

    B = np.zeros((parity_bits, parity_bits), dtype=np.int8)
    for row in range(parity_bits):
        B[row, row] = 1
        for offset in range(1, band_width + 1):
            if row - offset >= 0 and (offset == 1 or ((row + offset) % 2 == 0)):
                B[row, row - offset] = 1

    H = np.concatenate([A, B], axis=1)
    return H, A, B, codeword_bits, parity_bits

def build_edge_structure(H):
    check_count, variable_count = H.shape
    edge_variable = []
    check_edge_start = np.zeros(check_count + 1, dtype=np.int64)
    variable_neighbors = [[] for _ in range(variable_count)]
    edge_index = 0
    for check_index in range(check_count):
        variable_indices = np.where(H[check_index] == 1)[0]
        check_edge_start[check_index] = edge_index
        for variable_index in variable_indices:
            edge_variable.append(variable_index)
            variable_neighbors[variable_index].append(edge_index)
            edge_index += 1
    check_edge_start[check_count] = edge_index
    variable_edge_start = np.zeros(variable_count + 1, dtype=np.int64)
    variable_edges = np.zeros(edge_index, dtype=np.int64)
    fill_index = 0
    for variable_index in range(variable_count):
        variable_edge_start[variable_index] = fill_index
        for edge in variable_neighbors[variable_index]:
            variable_edges[fill_index] = edge
            fill_index += 1
    variable_edge_start[variable_count] = fill_index
    return np.asarray(edge_variable, dtype=np.int64), check_edge_start, variable_edges, variable_edge_start

def solve_lower_triangular_binary_system(B, rhs):
    n = len(rhs)
    solution = np.zeros(n, dtype=np.int8)
    for row in range(n):
        value = int(rhs[row])
        for col in range(row):
            if B[row, col] != 0:
                value ^= int(solution[col])
        solution[row] = value
    return solution

def ldpc_encode(information_bits, A, B):
    parity_count = A.shape[0]
    syndrome = np.zeros(parity_count, dtype=np.int8)
    for row in range(parity_count):
        value = 0
        for col in range(A.shape[1]):
            if A[row, col] != 0 and information_bits[col] != 0:
                value ^= 1
        syndrome[row] = value
    parity = solve_lower_triangular_binary_system(B, syndrome)
    return np.concatenate((information_bits, parity)).astype(np.int8)

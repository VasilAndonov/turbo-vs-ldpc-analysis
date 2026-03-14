import numpy as np
from ldpc.config import INFORMATION_BITS

def circulant_permutation_block(z_size, shift):
    block = np.zeros((z_size, z_size), dtype=np.int8)
    for row in range(z_size):
        block[row, (row + shift) % z_size] = 1
    return block

def get_qc_prototypes(rate_label):
    # H = [A | B], where B is lower-bidiagonal in circulant identity blocks.
    if rate_label == "1/3":
        a_proto = np.array([
            [0, 7],
            [13, 2],
            [5, 11],
            [9, 4],
        ], dtype=int)
    elif rate_label == "1/2":
        a_proto = np.array([
            [0, 9, 3, 12],
            [10, 2, 14, 6],
            [4, 13, 1, 11],
            [15, 7, 5, 8],
        ], dtype=int)
    elif rate_label == "3/4":
        a_proto = np.array([
            [0, 5, 9, 13, 17, 21, 25, 29, 1, 7, 11, 15],
            [3, 8, 12, 16, 20, 24, 28, 2, 6, 10, 14, 18],
            [19, 23, 27, 31, 4, 9, 13, 17, 21, 25, 29, 5],
            [6, 11, 15, 19, 23, 27, 30, 8, 12, 16, 20, 24],
        ], dtype=int)
    elif rate_label == "7/8":
        a_proto = np.array([
            [0, 1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12, 1, 3, 5, 7, 9, 11, 13, 2, 4, 6, 8, 10, 12, 14, 15],
            [2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9],
            [5, 7, 9, 11, 13, 15, 0, 4, 6, 8, 10, 12, 14, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15],
            [8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 6, 8, 10, 12, 14, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5],
        ], dtype=int)
    else:
        raise ValueError(rate_label)

    m_blocks = 4
    k_blocks = a_proto.shape[1]
    b_proto = -np.ones((m_blocks, m_blocks), dtype=int)
    for row in range(m_blocks):
        b_proto[row, row] = 0
        if row > 0:
            b_proto[row, row - 1] = 0
    return a_proto, b_proto

def build_qc_ldpc_matrices(rate_label):
    a_proto, b_proto = get_qc_prototypes(rate_label)
    m_blocks = a_proto.shape[0]
    k_blocks = a_proto.shape[1]

    if INFORMATION_BITS % k_blocks != 0:
        raise ValueError(f"INFORMATION_BITS={INFORMATION_BITS} is not divisible by k_blocks={k_blocks} for rate {rate_label}")

    z_size = INFORMATION_BITS // k_blocks
    parity_bits = m_blocks * z_size
    codeword_bits = INFORMATION_BITS + parity_bits

    h_rows = []
    for block_row in range(m_blocks):
        row_blocks = []
        for block_col in range(k_blocks):
            row_blocks.append(circulant_permutation_block(z_size, int(a_proto[block_row, block_col])))
        for block_col in range(m_blocks):
            shift = int(b_proto[block_row, block_col])
            if shift < 0:
                row_blocks.append(np.zeros((z_size, z_size), dtype=np.int8))
            else:
                row_blocks.append(circulant_permutation_block(z_size, shift))
        h_rows.append(np.concatenate(row_blocks, axis=1))

    h_matrix = np.concatenate(h_rows, axis=0)
    return h_matrix, a_proto, b_proto, z_size, codeword_bits, parity_bits

def build_edge_structure(h_matrix):
    check_count, variable_count = h_matrix.shape
    edge_variable = []
    check_edge_start = np.zeros(check_count + 1, dtype=np.int64)
    variable_neighbors = [[] for _ in range(variable_count)]

    edge_index = 0
    for check_index in range(check_count):
        variable_indices = np.where(h_matrix[check_index] == 1)[0]
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

def cyclic_shift(block, shift):
    return np.roll(block, shift, axis=0)

def encode_qc_ldpc(information_bits, a_proto, b_proto, z_size):
    information_bits = np.asarray(information_bits, dtype=np.int8)
    m_blocks = a_proto.shape[0]
    k_blocks = a_proto.shape[1]
    info_blocks = information_bits.reshape(k_blocks, z_size)

    syndrome_blocks = np.zeros((m_blocks, z_size), dtype=np.int8)
    for row in range(m_blocks):
        accum = np.zeros(z_size, dtype=np.int8)
        for col in range(k_blocks):
            accum ^= cyclic_shift(info_blocks[col], int(a_proto[row, col]))
        syndrome_blocks[row] = accum

    parity_blocks = np.zeros((m_blocks, z_size), dtype=np.int8)
    parity_blocks[0] = syndrome_blocks[0].copy()
    for row in range(1, m_blocks):
        parity_blocks[row] = syndrome_blocks[row] ^ parity_blocks[row - 1]

    return np.concatenate([information_bits, parity_blocks.reshape(-1)]).astype(np.int8)

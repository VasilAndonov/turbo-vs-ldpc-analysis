import numpy as np

from config import (
    RANDOM_SEED,
    INFORMATION_BIT_COUNT,
    PARITY_BIT_COUNT,
    CODEWORD_BIT_COUNT,
    INFORMATION_COLUMN_WEIGHT,
)

# ============================================================
# Parity-check matrix construction
# ============================================================
# The LDPC code is defined by a sparse parity-check matrix H.
# We build H in systematic-friendly form
#
#     H = [A | T]
#
# where:
#   - A connects parity checks to information bits
#   - T connects parity checks to parity bits
#
# T is chosen as a lower-bidiagonal matrix with ones on the main
# diagonal and the first subdiagonal. This keeps T invertible over
# GF(2), which makes encoding simple.


def build_information_connection_matrix():
    """Create the left sparse part A of the parity-check matrix.

    Each information-bit column gets a small, fixed number of ones.
    This preserves sparsity while making the graph connected enough
    for iterative decoding to be meaningful.
    """
    random_generator = np.random.default_rng(RANDOM_SEED)

    information_connection_matrix = np.zeros(
        (PARITY_BIT_COUNT, INFORMATION_BIT_COUNT), dtype=np.int8
    )

    for column_index in range(INFORMATION_BIT_COUNT):
        chosen_check_rows = random_generator.choice(
            PARITY_BIT_COUNT,
            size=INFORMATION_COLUMN_WEIGHT,
            replace=False,
        )
        information_connection_matrix[chosen_check_rows, column_index] = 1

    return information_connection_matrix



def build_parity_connection_matrix():
    """Create the right sparse part T of the parity-check matrix.

    T is lower-bidiagonal:
        1 on the diagonal,
        1 on the first subdiagonal.

    This means each parity bit mainly depends on the current parity
    equation and the previous parity bit. Because T is triangular with
    ones on the diagonal, it can be inverted by forward substitution in
    GF(2).
    """
    parity_connection_matrix = np.zeros(
        (PARITY_BIT_COUNT, PARITY_BIT_COUNT), dtype=np.int8
    )

    for row_index in range(PARITY_BIT_COUNT):
        parity_connection_matrix[row_index, row_index] = 1
        if row_index > 0:
            parity_connection_matrix[row_index, row_index - 1] = 1

    return parity_connection_matrix



def build_parity_check_matrix():
    """Build the full sparse parity-check matrix H."""
    information_connection_matrix = build_information_connection_matrix()
    parity_connection_matrix = build_parity_connection_matrix()
    parity_check_matrix = np.concatenate(
        [information_connection_matrix, parity_connection_matrix], axis=1
    )
    return (
        parity_check_matrix,
        information_connection_matrix,
        parity_connection_matrix,
    )


(
    PARITY_CHECK_MATRIX,
    INFORMATION_CONNECTION_MATRIX,
    PARITY_CONNECTION_MATRIX,
) = build_parity_check_matrix()

# ============================================================
# Graph view of the parity-check matrix
# ============================================================
# Iterative LDPC decoding works on the Tanner graph associated with H.
# We therefore precompute the check-node and variable-node neighbor lists.


def build_graph_from_parity_check_matrix(parity_check_matrix):
    """Convert H into neighbor lists for message passing."""
    check_to_variable_neighbors = []
    variable_to_check_neighbors = [list() for _ in range(parity_check_matrix.shape[1])]

    for check_index in range(parity_check_matrix.shape[0]):
        variable_neighbors = list(np.where(parity_check_matrix[check_index] == 1)[0])
        check_to_variable_neighbors.append(variable_neighbors)
        for variable_index in variable_neighbors:
            variable_to_check_neighbors[variable_index].append(check_index)

    return check_to_variable_neighbors, variable_to_check_neighbors


(
    CHECK_TO_VARIABLE_NEIGHBORS,
    VARIABLE_TO_CHECK_NEIGHBORS,
) = build_graph_from_parity_check_matrix(PARITY_CHECK_MATRIX)

# ============================================================
# LDPC encoding
# ============================================================
# A valid codeword c = [u | p] must satisfy H c^T = 0 over GF(2).
# With H = [A | T], we need:
#
#     A u + T p = 0   (mod 2)
#
# so the parity bits satisfy:
#
#     T p = A u       (mod 2)
#
# Because T is lower-bidiagonal, we can solve for p one element at a time.


def encode_information_bits(information_bits):
    """Encode an information block into a systematic LDPC codeword."""
    information_bits = np.asarray(information_bits, dtype=np.int8)
    if len(information_bits) != INFORMATION_BIT_COUNT:
        raise ValueError(
            f"Expected {INFORMATION_BIT_COUNT} information bits, got {len(information_bits)}."
        )

    syndrome_contribution = (
        INFORMATION_CONNECTION_MATRIX @ information_bits
    ) % 2

    parity_bits = np.zeros(PARITY_BIT_COUNT, dtype=np.int8)

    # Solve T p = A u over GF(2) by forward substitution.
    parity_bits[0] = syndrome_contribution[0]
    for row_index in range(1, PARITY_BIT_COUNT):
        parity_bits[row_index] = syndrome_contribution[row_index] ^ parity_bits[row_index - 1]

    codeword_bits = np.concatenate([information_bits, parity_bits])
    return codeword_bits.astype(np.int8)



def verify_codeword(codeword_bits):
    """Check whether a codeword satisfies all parity equations."""
    codeword_bits = np.asarray(codeword_bits, dtype=np.int8)
    syndrome = (PARITY_CHECK_MATRIX @ codeword_bits) % 2
    return bool(np.all(syndrome == 0))

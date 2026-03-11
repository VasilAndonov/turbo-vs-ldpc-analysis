import numpy as np
from encoder import NEXT_STATE_TABLE, PARITY_TABLE, SYSTEMATIC_SIGN_TABLE, PARITY_SIGN_TABLE, PREVIOUS_STATES, NEXT_STATES, TAIL_LENGTH, MEMORY_LENGTH

# -----------------------------
# Max-Log-MAP RSC decoder (terminated)
# -----------------------------
def rsc_max_log_map_terminated(systematic_llr, parity_llr, apriori_llr):
    """
    Max-log-MAP decoding of a terminated RSC encoder.
    Args:
        systematic_llr : Log-likelihood ratios of received systematic bits
        parity_llr     : Log-likelihood ratios of received parity bits
        apriori_llr    : Log-likelihood ratios from other decoder (extrinsic)
    Returns:
        posterior_llr : Posterior LLR for information bits
        extrinsic_llr : Extrinsic LLR (for iterative turbo decoding)
    """
    number_of_bits = len(systematic_llr)
    negative_infinity = -1e15

    # Initialize forward and backward metrics
    forward_metric = np.full((number_of_bits + 1, 4), negative_infinity, dtype=float)
    backward_metric = np.full((number_of_bits + 1, 4), negative_infinity, dtype=float)
    forward_metric[0, 0] = 0.0
    backward_metric[number_of_bits, 0] = 0.0

    # Branch metrics
    gamma0 = np.zeros((number_of_bits, 4), dtype=float)
    gamma1 = np.zeros((number_of_bits, 4), dtype=float)
    for bit_index in range(number_of_bits):
        for state in range(4):
            gamma0[bit_index, state] = 0.5 * ((systematic_llr[bit_index] + apriori_llr[bit_index]) * SYSTEMATIC_SIGN_TABLE[state, 0] +
                                              parity_llr[bit_index] * PARITY_SIGN_TABLE[state, 0])
            gamma1[bit_index, state] = 0.5 * ((systematic_llr[bit_index] + apriori_llr[bit_index]) * SYSTEMATIC_SIGN_TABLE[state, 1] +
                                              parity_llr[bit_index] * PARITY_SIGN_TABLE[state, 1])

    # Forward recursion
    for bit_index in range(number_of_bits):
        for next_state in range(4):
            best_metric = negative_infinity
            for previous_state, input_bit in PREVIOUS_STATES[next_state]:
                branch_metric = gamma0[bit_index, previous_state] if input_bit == 0 else gamma1[bit_index, previous_state]
                candidate_metric = forward_metric[bit_index, previous_state] + branch_metric
                if candidate_metric > best_metric:
                    best_metric = candidate_metric
            forward_metric[bit_index + 1, next_state] = best_metric

    # Backward recursion
    for bit_index in range(number_of_bits - 1, -1, -1):
        for state in range(4):
            best_metric = negative_infinity
            for next_state, input_bit in NEXT_STATES[state]:
                branch_metric = gamma0[bit_index, state] if input_bit == 0 else gamma1[bit_index, state]
                candidate_metric = backward_metric[bit_index + 1, next_state] + branch_metric
                if candidate_metric > best_metric:
                    best_metric = candidate_metric
            backward_metric[bit_index, state] = best_metric

    # Compute posterior LLR
    posterior_llr = np.zeros(number_of_bits, dtype=float)
    for bit_index in range(number_of_bits):
        best_zero = negative_infinity
        best_one = negative_infinity
        for state in range(4):
            next_state_zero = NEXT_STATE_TABLE[state, 0]
            next_state_one = NEXT_STATE_TABLE[state, 1]
            value_zero = forward_metric[bit_index, state] + gamma0[bit_index, state] + backward_metric[bit_index + 1, next_state_zero]
            value_one = forward_metric[bit_index, state] + gamma1[bit_index, state] + backward_metric[bit_index + 1, next_state_one]
            if value_zero > best_zero:
                best_zero = value_zero
            if value_one > best_one:
                best_one = value_one
        posterior_llr[bit_index] = best_zero - best_one

    # Compute extrinsic LLR
    extrinsic_llr = posterior_llr - systematic_llr - apriori_llr
    return posterior_llr, extrinsic_llr


def turbo_decode(systematic_1, parity_1, parity_2, interleaver, noise_variance, number_of_iterations):
    """
    Turbo decoding using two RSC decoders and iterative exchange.
    Args:
        systematic_1 : Received systematic bits (first encoder)
        parity_1     : Received parity bits from first encoder
        parity_2     : Received parity bits from second encoder
        interleaver  : Interleaver used in encoder
        noise_variance : Noise variance (sigma^2)
        number_of_iterations : Number of turbo iterations
    Returns:
        hard_decision_bits : Hard-decoded information bits
        llr_iterations    : LLR values at each iteration
    """
    total_length = len(systematic_1)
    deinterleaver = np.argsort(interleaver)
    channel_scaling_factor = 2.0 / noise_variance

    llr_systematic_1 = channel_scaling_factor * systematic_1
    llr_parity_1 = channel_scaling_factor * parity_1
    llr_parity_2 = channel_scaling_factor * parity_2

    apriori_1 = np.zeros(total_length, dtype=float)
    llr_iterations = []
    extrinsic_scaling_factor = 0.75

    permutation_full = np.concatenate([interleaver, np.arange(len(systematic_1), total_length)])
    inverse_permutation_full = np.argsort(permutation_full)

    for _ in range(number_of_iterations):
        _, extrinsic_1 = rsc_max_log_map_terminated(llr_systematic_1, llr_parity_1, apriori_1)
        apriori_2 = np.zeros(total_length, dtype=float)
        apriori_2[:len(interleaver)] = extrinsic_scaling_factor * extrinsic_1[interleaver]

        system_interleaved = np.zeros(total_length, dtype=float)
        system_interleaved[:len(interleaver)] = llr_systematic_1[interleaver]

        posterior_interleaved, extrinsic_interleaved = rsc_max_log_map_terminated(system_interleaved, llr_parity_2, apriori_2)

        posterior_info = np.zeros(len(interleaver), dtype=float)
        posterior_info[interleaver] = posterior_interleaved[:len(interleaver)]
        extrinsic_info = np.zeros(len(interleaver), dtype=float)
        extrinsic_info[interleaver] = extrinsic_interleaved[:len(interleaver)]

        llr_total = posterior_info.copy()
        llr_iterations.append(llr_total.copy())

        apriori_1[:len(interleaver)] = extrinsic_scaling_factor * extrinsic_info

    hard_decision_bits = (llr_iterations[-1] < 0.0).astype(np.int8)
    return hard_decision_bits, llr_iterations

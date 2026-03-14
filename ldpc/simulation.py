import time
import numpy as np
from ldpc.config import RANDOM_SEED, INFORMATION_BITS, ITERATIONS, LDPC_EBN0_DB, MIN_FRAMES, MAX_FRAMES, TARGET_ERRORS, BENCHMARK_BLOCKS, sigma2_from_ebn0, SELECTED_CODE_RATE_LABEL
from ldpc.encoder import build_ldpc_matrices, build_edge_structure, ldpc_encode
from ldpc.decoder import decode_ldpc_sum_product

def simulate_ldpc(rate_label=SELECTED_CODE_RATE_LABEL):
    rng = np.random.default_rng(RANDOM_SEED)
    H, A, B, codeword_bits, _ = build_ldpc_matrices(rate_label)
    edge_variable, check_edge_start, variable_edges, variable_edge_start = build_edge_structure(H)
    ber = {it: [] for it in ITERATIONS}
    llr_snapshot = {it: None for it in ITERATIONS}
    eff_rate = INFORMATION_BITS / codeword_bits
    for ebn0_db in LDPC_EBN0_DB:
        sigma2 = sigma2_from_ebn0(ebn0_db, eff_rate)
        sigma = np.sqrt(sigma2)
        errors = {it: 0 for it in ITERATIONS}
        bits_total = 0
        frames = 0
        while frames < MIN_FRAMES or (errors[max(ITERATIONS)] < TARGET_ERRORS and frames < MAX_FRAMES):
            bits = rng.integers(0, 2, INFORMATION_BITS, dtype=np.int8)
            codeword = ldpc_encode(bits, A, B)
            tx = 1.0 - 2.0 * codeword
            rx = tx + sigma * rng.standard_normal(codeword_bits)
            history = decode_ldpc_sum_product(rx, sigma2, max(ITERATIONS), H, check_edge_start, edge_variable, variable_edges, variable_edge_start)
            for it in ITERATIONS:
                info_llr = history[it-1][:INFORMATION_BITS]
                hard = (info_llr < 0.0).astype(np.int8)
                errors[it] += int(np.sum(bits != hard))
            if ebn0_db == LDPC_EBN0_DB[-1] and frames == 0:
                for it in ITERATIONS:
                    llr_snapshot[it] = history[it-1][:20].copy()
            bits_total += INFORMATION_BITS
            frames += 1
        for it in ITERATIONS:
            ber[it].append(errors[it] / max(bits_total, 1))
    for it in ITERATIONS:
        ber[it] = np.array(ber[it], dtype=float)
    return ber, llr_snapshot

def benchmark_ldpc(rate_label=SELECTED_CODE_RATE_LABEL):
    rng = np.random.default_rng(RANDOM_SEED + 100)
    H, A, B, codeword_bits, _ = build_ldpc_matrices(rate_label)
    edge_variable, check_edge_start, variable_edges, variable_edge_start = build_edge_structure(H)
    eff_rate = INFORMATION_BITS / codeword_bits
    sigma2 = sigma2_from_ebn0(0.5, eff_rate)
    sigma = np.sqrt(sigma2)
    timings = {}
    for it in ITERATIONS:
        start = time.perf_counter()
        for _ in range(BENCHMARK_BLOCKS):
            bits = rng.integers(0, 2, INFORMATION_BITS, dtype=np.int8)
            codeword = ldpc_encode(bits, A, B)
            tx = 1.0 - 2.0 * codeword
            rx = tx + sigma * rng.standard_normal(codeword_bits)
            decode_ldpc_sum_product(rx, sigma2, it, H, check_edge_start, edge_variable, variable_edges, variable_edge_start)
        timings[it] = time.perf_counter() - start
    return timings

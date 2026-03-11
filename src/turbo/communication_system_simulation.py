import numpy as np
from configuration_parameters import *
from convolutional_and_turbo_encoders import *
from viterbi_and_turbo_decoders import *

# ============================================================
# NOISE VARIANCE
# ============================================================

def compute_noise_variance_from_ebn0(ebn0_db, code_rate):
    ebn0_linear = 10**(ebn0_db/10)
    return 1.0 / (2.0*code_rate*ebn0_linear)

# ============================================================
# CONVOLUTIONAL SIMULATION
# ============================================================

def run_convolutional_code_simulation(random_generator):
    bit_error_rates = []
    for ebn0 in CONVOLUTIONAL_EBN0_VALUES_DB:
        sigma2 = compute_noise_variance_from_ebn0(ebn0,0.5)
        sigma = np.sqrt(sigma2)
        bit_errors = 0
        total_bits = 0
        frames = 0
        while frames < CONVOLUTIONAL_MINIMUM_FRAMES:
            info_bits = random_generator.integers(0,2,INFORMATION_BLOCK_LENGTH,dtype=np.int8)
            encoded_bits = convolutional_encode_generator_7_5(info_bits)
            tx = 1.0 - 2.0*encoded_bits
            rx = tx + sigma*random_generator.standard_normal(len(tx))
            decoded_bits = viterbi_decode_convolutional_code(rx, INFORMATION_BLOCK_LENGTH)
            bit_errors += np.sum(info_bits != decoded_bits)
            total_bits += INFORMATION_BLOCK_LENGTH
            frames += 1
        ber = bit_errors/total_bits
        bit_error_rates.append(ber)
        print(f"Eb/N0={ebn0} dB, BER={ber:.4e}")
    return np.array(bit_error_rates)

# ============================================================
# TURBO SIMULATION
# ============================================================

def run_turbo_code_simulation(random_generator):
    interleaver = random_generator.permutation(INFORMATION_BLOCK_LENGTH)
    turbo_results = {it: [] for it in TURBO_DECODER_ITERATION_COUNTS}
    llr_snapshot = {it: None for it in TURBO_DECODER_ITERATION_COUNTS}

    for ebn0 in TURBO_EBN0_VALUES_DB:
        sigma2 = compute_noise_variance_from_ebn0(ebn0, INFORMATION_BLOCK_LENGTH/(3*INFORMATION_BLOCK_LENGTH+TAIL_BIT_COUNT))
        sigma = np.sqrt(sigma2)
        bit_errors = {it: 0 for it in TURBO_DECODER_ITERATION_COUNTS}
        n_bits = 0
        frames = 0

        while frames < TURBO_MINIMUM_FRAMES:
            info_bits = random_generator.integers(0,2,INFORMATION_BLOCK_LENGTH,dtype=np.int8)
            sys1, par1, sys2, par2 = turbo_encode_information_bits(info_bits, interleaver)
            rx_sys1 = 1.0-2.0*sys1 + sigma*random_generator.standard_normal(len(sys1))
            rx_par1 = 1.0-2.0*par1 + sigma*random_generator.standard_normal(len(par1))
            rx_par2 = 1.0-2.0*par2 + sigma*random_generator.standard_normal(len(par2))
            hard_bits, llr_iters = turbo_maxlog_map_decode(rx_sys1, rx_par1, rx_par2, interleaver, sigma2, max(TURBO_DECODER_ITERATION_COUNTS))
            for it in TURBO_DECODER_ITERATION_COUNTS:
                bit_errors[it] += np.sum(info_bits != hard_bits)
            if frames == 0:
                for it in TURBO_DECODER_ITERATION_COUNTS:
                    llr_snapshot[it] = llr_iters[it-1][:20].copy()
            n_bits += INFORMATION_BLOCK_LENGTH
            frames += 1

        for it in TURBO_DECODER_ITERATION_COUNTS:
            turbo_results[it].append(bit_errors[it]/n_bits)

    for it in TURBO_DECODER_ITERATION_COUNTS:
        turbo_results[it] = np.array(turbo_results[it])

    return turbo_results, llr_snapshot

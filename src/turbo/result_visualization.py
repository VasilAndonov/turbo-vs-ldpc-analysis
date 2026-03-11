import matplotlib.pyplot as plt
from scipy.special import erfc
from configuration_parameters import *

def plot_all_simulation_results(convolutional_ber, turbo_results, llr_snapshot):
    floor = 1e-8

    # 1 Convolutional BER
    plt.figure(figsize=(7,5))
    uncoded = 0.5*erfc(np.sqrt(10**(CONVOLUTIONAL_EBN0_VALUES_DB/10)))
    plt.semilogy(CONVOLUTIONAL_EBN0_VALUES_DB, np.clip(uncoded,floor,None), 'r.-', label="Uncoded BPSK")
    plt.semilogy(CONVOLUTIONAL_EBN0_VALUES_DB, np.clip(convolutional_ber,floor,None), 'bo-', label="Convolutional (7,5)")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("Convolutional Code Performance")
    plt.grid(True, which='both')
    plt.legend()
    plt.show()

    # 2 Turbo BER per iteration
    plt.figure(figsize=(7,5))
    for it in TURBO_DECODER_ITERATION_COUNTS:
        plt.semilogy(TURBO_EBN0_VALUES_DB, np.clip(turbo_results[it], floor,None), marker='x', label=f'Iteration {it}')
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("BER")
    plt.title("Turbo Code BER per Iteration")
    plt.grid(True, which='both')
    plt.legend()
    plt.show()

    # 3 Turbo LLR growth
    plt.figure(figsize=(7,5))
    for it in TURBO_DECODER_ITERATION_COUNTS:
        if llr_snapshot[it] is not None:
            plt.scatter(range(len(llr_snapshot[it])), llr_snapshot[it], label=f'Iteration {it}')
    plt.xlabel("Bit index")
    plt.ylabel("Posterior LLR")
    plt.title("Turbo Decoder LLR Growth")
    plt.grid(True)
    plt.legend()
    plt.show()

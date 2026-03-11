import matplotlib.pyplot as plt
from scipy.special import erfc
from simulation import CONV_EBN0_DB, TURBO_EBN0_DB, ITERATIONS

SAVE_FIGURES = False
SAVE_PREFIX = "turbo_lit"

def plot_all(conv_ber, turbo_results, llr_snapshot):
    floor = 1e-8

    # Convolutional baseline
    plt.figure(figsize=(7, 4.8))
    uncoded = 0.5 * erfc(np.sqrt(10.0 ** (CONV_EBN0_DB / 10.0)))
    plt.semilogy(CONV_EBN0_DB, np.clip(uncoded, floor, None), 'r.-', label='Uncoded BPSK')
    plt.semilogy(CONV_EBN0_DB, np.clip(conv_ber, floor, None), 'bo-', label='Conv. code (7,5)')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.title('Convolutional baseline')
    plt.grid(True, which='both', alpha=0.35)
    plt.legend()
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(f'{SAVE_PREFIX}_conv.png', dpi=160)
    plt.show()

    # Turbo BER
    plt.figure(figsize=(7, 4.8))
    for it in ITERATIONS:
        plt.semilogy(TURBO_EBN0_DB, np.clip(turbo_results[it], floor, None), marker='x', label=f'Iteration {it}')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.title('Turbo code BER (terminated RSC, max-log-MAP)')
    plt.grid(True, which='both', alpha=0.35)
    plt.legend()
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(f'{SAVE_PREFIX}_turbo.png', dpi=160)
    plt.show()

    # LLR growth
    plt.figure(figsize=(8, 5.2))
    for it in ITERATIONS:
        if llr_snapshot[it] is not None:
            plt.scatter(range(len(llr_snapshot[it])), llr_snapshot[it], s=18, label=f'Iteration {it}')
    plt.xlabel('Bit index')
    plt.ylabel('Posterior LLR')
    plt.title('Turbo decoder confidence growth')
    plt.grid(True, linestyle='--', alpha=0.45)
    plt.legend()
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(f'{SAVE_PREFIX}_llr.png', dpi=160)
    plt.show()

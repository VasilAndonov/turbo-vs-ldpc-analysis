import numpy as np
import matplotlib.pyplot as plt
from simulation import monte_carlo_turbo
from encoder import turbo_encode

# -----------------------------
# Configuration
# -----------------------------
RANDOM_SEED = 12
INFORMATION_LENGTH = 1024
ITERATIONS_LIST = [1, 2, 3, 4, 5, 6]
TURBO_EBN0_DB_ARRAY = np.arange(-1.0, 1.76, 0.25)
SAVE_FIGURES = False
SAVE_PREFIX = 'turbo_simulation'

# -----------------------------
# Run simulation
# -----------------------------
def main():
    rng = np.random.default_rng(RANDOM_SEED)
    interleaver = rng.permutation(INFORMATION_LENGTH)

    turbo_results, llr_snapshot = monte_carlo_turbo(rng, INFORMATION_LENGTH, interleaver,
                                                    TURBO_EBN0_DB_ARRAY, ITERATIONS_LIST)

    # -----------------------------
    # Plot Turbo BER
    # -----------------------------
    plt.figure(figsize=(7, 4.8))
    floor_value = 1e-8
    for iteration in ITERATIONS_LIST:
        plt.semilogy(TURBO_EBN0_DB_ARRAY, np.clip(turbo_results[iteration], floor_value, None),
                     marker='x', label=f'Iteration {iteration}')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('Turbo Code BER vs Eb/N0')
    plt.grid(True, which='both', alpha=0.35)
    plt.legend()
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(f'{SAVE_PREFIX}_turbo_ber.png', dpi=160)
    plt.show()

    # -----------------------------
    # Plot LLR growth
    # -----------------------------
    plt.figure(figsize=(8, 5.2))
    for iteration in ITERATIONS_LIST:
        if llr_snapshot[iteration] is not None:
            plt.scatter(np.arange(len(llr_snapshot[iteration])),
                        llr_snapshot[iteration], s=18, label=f'Iteration {iteration}')
    plt.xlabel('Bit Index')
    plt.ylabel('Posterior LLR')
    plt.title('Turbo Decoder Confidence Growth')
    plt.grid(True, linestyle='--', alpha=0.45)
    plt.legend()
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(f'{SAVE_PREFIX}_llr_growth.png', dpi=160)
    plt.show()


if __name__ == '__main__':
    main()

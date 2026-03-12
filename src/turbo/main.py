import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

from config import (
    CONVOLUTIONAL_EB_NO_DB,
    DECODER_ITERATION_LIST,
    RANDOM_SEED,
    SAVE_FIGURES,
    SAVE_PREFIX,
    TURBO_EB_NO_DB,
    get_rate_label,
)
from simulation import run_convolutional_simulation, run_turbo_simulation

def plot_results(convolutional_ber, turbo_ber_by_iteration, llr_snapshot):
    floor_value = 1e-8

    plt.figure(figsize=(7.0, 4.8))
    uncoded_curve = 0.5 * erfc(np.sqrt(10.0 ** (CONVOLUTIONAL_EB_NO_DB / 10.0)))
    plt.semilogy(CONVOLUTIONAL_EB_NO_DB, np.clip(uncoded_curve, floor_value, None), "r.-", label="Uncoded BPSK")
    plt.semilogy(CONVOLUTIONAL_EB_NO_DB, np.clip(convolutional_ber, floor_value, None), "bo-", label="Convolutional baseline")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Bit error rate")
    plt.title("Convolutional baseline")
    plt.grid(True, which="both", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(f"{SAVE_PREFIX}_conv.png", dpi=170)
    plt.show()

    plt.figure(figsize=(7.0, 4.8))
    for iteration_count in DECODER_ITERATION_LIST:
        plt.semilogy(
            TURBO_EB_NO_DB,
            np.clip(turbo_ber_by_iteration[iteration_count], floor_value, None),
            marker="x",
            label=f"Iteration {iteration_count}",
        )
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Bit error rate")
    plt.title(f"Turbo code BER, rate {get_rate_label()}")
    plt.grid(True, which="both", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(f"{SAVE_PREFIX}_turbo_{get_rate_label().replace('/', '_')}.png", dpi=170)
    plt.show()

    plt.figure(figsize=(8.0, 5.0))
    for iteration_count in DECODER_ITERATION_LIST:
        if llr_snapshot[iteration_count] is not None:
            plt.scatter(range(len(llr_snapshot[iteration_count])), llr_snapshot[iteration_count], s=18, label=f"Iteration {iteration_count}")
    plt.xlabel("Bit index")
    plt.ylabel("Posterior LLR")
    plt.title(f"Turbo decoder confidence growth, rate {get_rate_label()}")
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend()
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(f"{SAVE_PREFIX}_llr_{get_rate_label().replace('/', '_')}.png", dpi=170)
    plt.show()

def main():
    random_generator = np.random.default_rng(RANDOM_SEED)
    start_time = time.time()
    convolutional_ber = run_convolutional_simulation(random_generator)
    turbo_ber_by_iteration, llr_snapshot = run_turbo_simulation(random_generator)
    plot_results(convolutional_ber, turbo_ber_by_iteration, llr_snapshot)
    print(f"Turbo simulation finished in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()

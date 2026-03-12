"""
Main entry point for the optimized LDPC project.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from config import DECODER_ITERATION_LIST, LDPC_EB_NO_DB, RANDOM_SEED, SAVE_FIGURES, SAVE_PREFIX
from simulation import run_ldpc_simulation


def plot_ldpc_results(ber_by_iteration, llr_snapshot):
    floor_value = 1e-8

    plt.figure(figsize=(7.2, 4.8))
    for iteration_count in DECODER_ITERATION_LIST:
        plt.semilogy(
            LDPC_EB_NO_DB,
            np.clip(ber_by_iteration[iteration_count], floor_value, None),
            marker="o",
            label=f"Iteration {iteration_count}",
        )

    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Bit error rate")
    plt.title("Optimized LDPC code BER")
    plt.grid(True, which="both", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    if SAVE_FIGURES:
        plt.savefig(f"{SAVE_PREFIX}_ber.png", dpi=170)
    plt.show()

    plt.figure(figsize=(8.0, 5.0))
    for iteration_count in DECODER_ITERATION_LIST:
        if llr_snapshot[iteration_count] is not None:
            plt.scatter(
                range(len(llr_snapshot[iteration_count])),
                llr_snapshot[iteration_count],
                s=18,
                label=f"Iteration {iteration_count}",
            )

    plt.xlabel("Bit index")
    plt.ylabel("Posterior LLR")
    plt.title("LDPC decoder confidence growth")
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend()
    plt.tight_layout()

    if SAVE_FIGURES:
        plt.savefig(f"{SAVE_PREFIX}_llr.png", dpi=170)
    plt.show()


def main():
    random_generator = np.random.default_rng(RANDOM_SEED)
    start_time = time.time()

    ber_by_iteration, llr_snapshot = run_ldpc_simulation(random_generator)
    plot_ldpc_results(ber_by_iteration, llr_snapshot)

    elapsed_seconds = time.time() - start_time
    print(f"Optimized LDPC simulation finished in {elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    main()

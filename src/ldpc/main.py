import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import (
    EBN0_DECIBELS,
    ITERATION_LIST,
    SAVE_FIGURES,
    SAVE_PREFIX,
)
from simulation import run_ldpc_simulation

# ============================================================
# Plotting
# ============================================================
# The plot style is intentionally close to the turbo-code project so that
# both sets of results can be compared side by side in a report.


def plot_ldpc_results(bit_error_rate_results, llr_snapshot_by_iteration):
    floor_value = 1e-8

    plt.figure(figsize=(7, 4.8))
    for iteration_count in ITERATION_LIST:
        clipped_curve = np.clip(bit_error_rate_results[iteration_count], floor_value, None)
        plt.semilogy(
            EBN0_DECIBELS,
            clipped_curve,
            marker="x",
            label=f"Iteration {iteration_count}",
        )
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("BER")
    plt.title("LDPC code BER (normalized min-sum decoding)")
    plt.grid(True, which="both", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(f"{SAVE_PREFIX}_ber.png", dpi=160)
    plt.show()

    plt.figure(figsize=(8, 5.2))
    for iteration_count in ITERATION_LIST:
        if llr_snapshot_by_iteration[iteration_count] is not None:
            plt.scatter(
                range(len(llr_snapshot_by_iteration[iteration_count])),
                llr_snapshot_by_iteration[iteration_count],
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
        plt.savefig(f"{SAVE_PREFIX}_llr.png", dpi=160)
    plt.show()


# ============================================================
# Entry point
# ============================================================
# This file ties the simulation and the plots together so the project can
# be run with a single command.


def main():
    start_time = time.time()
    bit_error_rate_results, llr_snapshot_by_iteration = run_ldpc_simulation()
    plot_ldpc_results(bit_error_rate_results, llr_snapshot_by_iteration)
    print(f"Simulation done in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()

"""Main entry point for the turbo-code BER simulation.

Run this file to generate the convolutional baseline, the turbo BER curves, and the LLR
confidence plot.
"""

import time

import numpy as np

from config import RANDOM_SEED
from simulation import (
    plot_simulation_results,
    run_convolutional_simulation,
    run_turbo_simulation,
)


# --------------------------------------------------------------------------------------
# Main experiment flow
# --------------------------------------------------------------------------------------
# The same random generator is shared across the whole run so the experiment remains
# reproducible.

def main():
    random_number_generator = np.random.default_rng(RANDOM_SEED)
    start_time = time.time()

    convolutional_bit_error_rate = run_convolutional_simulation(random_number_generator)
    turbo_bit_error_rate_by_iteration, llr_snapshot_by_iteration = run_turbo_simulation(random_number_generator)
    plot_simulation_results(
        convolutional_bit_error_rate,
        turbo_bit_error_rate_by_iteration,
        llr_snapshot_by_iteration,
    )

    print(f"Simulation finished in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()

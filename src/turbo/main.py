import numpy as np
import time
from configuration_parameters import RANDOM_SEED
from communication_system_simulation import run_convolutional_code_simulation, run_turbo_code_simulation
from result_visualization import plot_all_simulation_results

def main():
    rng = np.random.default_rng(RANDOM_SEED)
    start_time = time.time()
    print("Starting simulation...")

    # Run convolutional code
    convolutional_ber = run_convolutional_code_simulation(rng)

    # Run turbo code
    turbo_results, llr_snapshot = run_turbo_code_simulation(rng)

    # Plot all results
    plot_all_simulation_results(convolutional_ber, turbo_results, llr_snapshot)

    print(f"Simulation finished in {time.time()-start_time:.2f} seconds")

if __name__ == "__main__":
    main()

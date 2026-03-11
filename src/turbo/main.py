import numpy as np
import time
from simulation import run_conv, run_turbo
from plotting import plot_all

SEED = 12

def main():
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    print("Running Turbo code simulation...")

    conv_ber = run_conv(rng)
    turbo_results, llr_snapshot = run_turbo(rng)
    plot_all(conv_ber, turbo_results, llr_snapshot)

    print(f"Simulation done in {time.time() - t0:.2f} seconds")

if __name

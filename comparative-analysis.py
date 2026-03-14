import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

from turbo.simulation import simulate_turbo, benchmark_turbo
from turbo.config import ITERATIONS as TURBO_ITERATIONS, TURBO_EBN0_DB, INFORMATION_BITS as TURBO_INFO_BITS, BENCHMARK_BLOCKS as TURBO_BENCH, SHOW_PLOTS, SAVE_PLOTS
from ldpc.simulation import simulate_ldpc, benchmark_ldpc
from ldpc.config import ITERATIONS as LDPC_ITERATIONS, LDPC_EBN0_DB, BENCHMARK_BLOCKS as LDPC_BENCH

CODE_RATE_LABELS = ["1/3", "1/2", "3/4", "7/8"]
PLOT_PREFIX = "turbo_vs_ldpc"

def plot_ber_by_rate(results_turbo, results_ldpc):
    floor = 1e-8
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    axes = axes.ravel()

    for idx, rate_label in enumerate(CODE_RATE_LABELS):
        ax = axes[idx]
        ax.set_title(f"Code rate {rate_label}")

        uncoded = 0.5 * erfc(np.sqrt(10.0 ** (TURBO_EBN0_DB / 10.0)))
        ax.semilogy(TURBO_EBN0_DB, np.clip(uncoded, floor, None), "r.-", label="Uncoded BPSK")
        ax.semilogy(TURBO_EBN0_DB, np.clip(results_turbo[rate_label][max(TURBO_ITERATIONS)], floor, None), "ko-", label=f"Turbo, iteration {max(TURBO_ITERATIONS)}")
        ax.semilogy(LDPC_EBN0_DB, np.clip(results_ldpc[rate_label][max(LDPC_ITERATIONS)], floor, None), color="tab:green", marker="D", label=f"LDPC, iteration {max(LDPC_ITERATIONS)}")

        ax.grid(True, which="both", alpha=0.35)
        ax.set_xlabel("Eb/N0 (dB)")
        ax.set_ylabel("BER")
        ax.legend(fontsize=8)

    fig.suptitle("BER versus SNR for Turbo and LDPC")
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f"{PLOT_PREFIX}_ber_by_rate.png", dpi=180)
    if SHOW_PLOTS:
        plt.show()

def plot_throughput(runtime_turbo, runtime_ldpc):
    turbo_it = np.array(sorted(runtime_turbo.keys()))
    ldpc_it = np.array(sorted(runtime_ldpc.keys()))
    turbo_time = np.array([runtime_turbo[it] for it in turbo_it], dtype=float)
    ldpc_time = np.array([runtime_ldpc[it] for it in ldpc_it], dtype=float)
    turbo_bps = (TURBO_INFO_BITS * TURBO_BENCH) / np.maximum(turbo_time, 1e-12)
    ldpc_bps = (TURBO_INFO_BITS * LDPC_BENCH) / np.maximum(ldpc_time, 1e-12)

    plt.figure(figsize=(11.0, 4.8))

    plt.subplot(1, 2, 1)
    plt.plot(turbo_it, turbo_time, "o-", label="Turbo runtime")
    plt.plot(ldpc_it, ldpc_time, "s--", label="LDPC runtime")
    plt.xlabel("Maximum decoder iteration count")
    plt.ylabel("Elapsed time (seconds)")
    plt.title("Decoder runtime")
    plt.grid(True, alpha=0.35)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(turbo_it, turbo_bps, "o-", label="Turbo bits/s")
    plt.plot(ldpc_it, ldpc_bps, "s--", label="LDPC bits/s")
    plt.xlabel("Maximum decoder iteration count")
    plt.ylabel("Decoded information bits per second")
    plt.title("Decoder throughput")
    plt.grid(True, alpha=0.35)
    plt.legend()

    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f"{PLOT_PREFIX}_throughput.png", dpi=180)
    if SHOW_PLOTS:
        plt.show()

def main():
    turbo_results = {}
    ldpc_results = {}
    for rate_label in CODE_RATE_LABELS:
        turbo_ber, _ = simulate_turbo(rate_label)
        ldpc_ber, _ = simulate_ldpc(rate_label)
        turbo_results[rate_label] = turbo_ber
        ldpc_results[rate_label] = ldpc_ber
    plot_ber_by_rate(turbo_results, ldpc_results)
    plot_throughput(benchmark_turbo("1/3"), benchmark_ldpc("1/3"))

if __name__ == "__main__":
    main()

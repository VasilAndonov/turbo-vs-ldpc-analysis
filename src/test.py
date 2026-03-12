from __future__ import annotations

"""
Rate-and-throughput comparison script for the turbo and LDPC projects.

What this script tries to do
----------------------------
1) BER vs SNR grouped by code rate:
   It tries to run both projects at rates 1/2, 1/3, 3/4, and 7/8.
   For each supported rate, it plots the best available turbo and LDPC BER curves.

2) Throughput / iteration-complexity view:
   It times the simulation runs while varying the maximum decoder iteration count,
   then plots wall-clock runtime and a simple relative throughput proxy.

Important note
--------------
This file can only compare rates that the underlying turbo and LDPC projects
actually support. If one repository cannot be configured for a requested rate,
the script reports that and skips that rate instead of crashing.

Expected folder layout
----------------------
src/
├── turbo/
│   ├── config.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── simulation.py
│   └── main.py
├── ldpc/
│   ├── config.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── simulation.py
│   └── main.py
└── turbo_vs_ldpc_rates_and_throughput.py
"""

import importlib
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc


# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------
TURBO_FOLDER = Path("turbo")
LDPC_FOLDER = Path("ldpc")
OUTPUT_FOLDER = Path("comparison_results")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

CODE_RATES_TO_COMPARE = [1 / 2, 1 / 3, 3 / 4, 7 / 8]
THROUGHPUT_ITERATION_LIST = [1, 2, 3, 4, 5, 6]

SAVE_FIGURES = True
SHOW_FIGURES = True
FIGURE_PREFIX = "turbo_vs_ldpc_rates"


# ---------------------------------------------------------------------
# Import isolation helpers
# ---------------------------------------------------------------------
PROJECT_MODULE_NAMES = ["config", "encoder", "decoder", "simulation", "main"]


def clear_project_modules():
    """Remove one project's local modules before importing the next one."""
    for module_name in PROJECT_MODULE_NAMES:
        sys.modules.pop(module_name, None)


class TemporaryProjectPath:
    """Temporarily add one project folder to sys.path."""

    def __init__(self, folder: Path):
        self.folder = str(folder.resolve())

    def __enter__(self):
        sys.path.insert(0, self.folder)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if sys.path and sys.path[0] == self.folder:
            sys.path.pop(0)
        else:
            try:
                sys.path.remove(self.folder)
            except ValueError:
                pass
        clear_project_modules()


def get_first_existing_attribute(module, candidate_names, label):
    """Return the first matching attribute name from a list of candidates."""
    for name in candidate_names:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"Could not find any of {candidate_names} in {label}")


def try_set_first_existing_attribute(module, candidate_names, value):
    """Set the first matching attribute and return True if successful."""
    for name in candidate_names:
        if hasattr(module, name):
            setattr(module, name, value)
            return True
    return False


def normalize_iteration_results(raw_results, iteration_list):
    """Convert result containers to a dictionary keyed by iteration count."""
    if isinstance(raw_results, dict):
        return {
            int(iteration_count): np.array(raw_results[iteration_count], dtype=float)
            for iteration_count in iteration_list
        }

    if isinstance(raw_results, (list, tuple)):
        if len(raw_results) != len(iteration_list):
            raise ValueError("Result container length does not match iteration list length.")
        return {
            int(iteration_list[index]): np.array(raw_results[index], dtype=float)
            for index in range(len(iteration_list))
        }

    raise TypeError("Unsupported BER result container type.")


def format_rate_label(code_rate: float) -> str:
    """Return a readable label for a code rate."""
    mapping = {
        1 / 2: "1/2",
        1 / 3: "1/3",
        3 / 4: "3/4",
        7 / 8: "7/8",
    }
    return mapping.get(code_rate, f"{code_rate:.3f}")


def uncoded_bpsk_ber(ebn0_db_values):
    """Theoretical BER of uncoded BPSK in AWGN."""
    ebn0_db_values = np.asarray(ebn0_db_values, dtype=float)
    return 0.5 * erfc(np.sqrt(10.0 ** (ebn0_db_values / 10.0)))


def finish_figure(file_name: str):
    """Save and/or show the current matplotlib figure."""
    output_path = OUTPUT_FOLDER / file_name
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(output_path, dpi=180)
        print(f"Saved: {output_path}")
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------------
# Configuration hooks
# ---------------------------------------------------------------------
def apply_requested_rate(config_module, requested_rate: float) -> bool:
    """
    Try to configure a project to use a requested code rate.

    This function supports a few likely naming styles. If the project does not
    expose a configurable rate, the function returns False.
    """
    for setter_name in ["set_code_rate", "configure_code_rate", "apply_code_rate"]:
        if hasattr(config_module, setter_name):
            getattr(config_module, setter_name)(requested_rate)
            return True

    for variable_name in [
        "CODE_RATE",
        "NOMINAL_CODE_RATE",
        "TURBO_CODE_RATE",
        "LDPC_CODE_RATE",
        "RATE",
    ]:
        if hasattr(config_module, variable_name):
            setattr(config_module, variable_name, requested_rate)
            return True

    return False


def apply_requested_iteration_list(config_module, iteration_list) -> bool:
    """Try to update the decoder iteration list exposed by one project."""
    for variable_name in [
        "DECODER_ITERATION_LIST",
        "ITERATION_LIST",
        "ITERATIONS",
    ]:
        if hasattr(config_module, variable_name):
            setattr(config_module, variable_name, list(iteration_list))
            return True
    return False


# ---------------------------------------------------------------------
# Turbo project driver
# ---------------------------------------------------------------------
def run_turbo_project_for_rate(code_rate: float):
    """
    Run the turbo project at one requested code rate.

    The call returns None when the project cannot be configured for that rate.
    """
    if not TURBO_FOLDER.exists():
        raise FileNotFoundError(f"Turbo folder not found: {TURBO_FOLDER}")

    with TemporaryProjectPath(TURBO_FOLDER):
        turbo_config = importlib.import_module("config")
        rate_supported = apply_requested_rate(turbo_config, code_rate)

        if not rate_supported and code_rate != 1 / 3:
            print(f"Turbo project does not expose a rate hook for {format_rate_label(code_rate)}. Skipping.")
            return None

        turbo_simulation = importlib.import_module("simulation")

        random_seed = getattr(turbo_config, "RANDOM_SEED", 12)
        random_generator = np.random.default_rng(random_seed)

        run_convolutional_simulation = get_first_existing_attribute(
            turbo_simulation,
            ["run_convolutional_simulation", "run_conv"],
            "turbo.simulation",
        )
        run_turbo_simulation = get_first_existing_attribute(
            turbo_simulation,
            ["run_turbo_simulation", "run_turbo"],
            "turbo.simulation",
        )

        convolutional_ebn0_db = np.array(
            get_first_existing_attribute(
                turbo_simulation,
                ["CONVOLUTIONAL_EB_NO_DB", "CONV_EBN0_DB"],
                "turbo.simulation",
            ),
            dtype=float,
        )
        turbo_ebn0_db = np.array(
            get_first_existing_attribute(
                turbo_simulation,
                ["TURBO_EB_NO_DB", "TURBO_EBN0_DB"],
                "turbo.simulation",
            ),
            dtype=float,
        )
        turbo_iteration_list = list(
            get_first_existing_attribute(
                turbo_config,
                ["DECODER_ITERATION_LIST", "ITERATIONS"],
                "turbo.config",
            )
        )

        start_time = time.perf_counter()
        convolutional_ber = run_convolutional_simulation(random_generator)
        turbo_results, turbo_llr_snapshot = run_turbo_simulation(random_generator)
        elapsed_seconds = time.perf_counter() - start_time

        information_length = getattr(
            turbo_config,
            "INFORMATION_BLOCK_LENGTH",
            getattr(turbo_config, "K", None),
        )

        return {
            "rate": code_rate,
            "convolutional_ebn0_db": convolutional_ebn0_db,
            "convolutional_ber": np.array(convolutional_ber, dtype=float),
            "turbo_ebn0_db": turbo_ebn0_db,
            "iteration_list": turbo_iteration_list,
            "ber_by_iteration": normalize_iteration_results(turbo_results, turbo_iteration_list),
            "llr_snapshot": turbo_llr_snapshot,
            "information_length": information_length,
            "elapsed_seconds": elapsed_seconds,
        }


def benchmark_turbo_iterations(base_rate: float = 1 / 3):
    """Measure turbo runtime as the maximum decoder iteration count changes."""
    if not TURBO_FOLDER.exists():
        raise FileNotFoundError(f"Turbo folder not found: {TURBO_FOLDER}")

    measurements = {}

    for iteration_count in THROUGHPUT_ITERATION_LIST:
        with TemporaryProjectPath(TURBO_FOLDER):
            turbo_config = importlib.import_module("config")
            apply_requested_rate(turbo_config, base_rate)
            apply_requested_iteration_list(turbo_config, list(range(1, iteration_count + 1)))

            turbo_simulation = importlib.import_module("simulation")
            random_seed = getattr(turbo_config, "RANDOM_SEED", 12)
            random_generator = np.random.default_rng(random_seed)

            run_turbo_simulation = get_first_existing_attribute(
                turbo_simulation,
                ["run_turbo_simulation", "run_turbo"],
                "turbo.simulation",
            )

            start_time = time.perf_counter()
            run_turbo_simulation(random_generator)
            elapsed_seconds = time.perf_counter() - start_time
            measurements[iteration_count] = elapsed_seconds

    return measurements


# ---------------------------------------------------------------------
# LDPC project driver
# ---------------------------------------------------------------------
def run_ldpc_project_for_rate(code_rate: float):
    """
    Run the LDPC project at one requested code rate.

    The call returns None when the project cannot be configured for that rate.
    """
    if not LDPC_FOLDER.exists():
        raise FileNotFoundError(f"LDPC folder not found: {LDPC_FOLDER}")

    with TemporaryProjectPath(LDPC_FOLDER):
        ldpc_config = importlib.import_module("config")
        rate_supported = apply_requested_rate(ldpc_config, code_rate)

        if not rate_supported and code_rate != 1 / 3:
            print(f"LDPC project does not expose a rate hook for {format_rate_label(code_rate)}. Skipping.")
            return None

        ldpc_simulation = importlib.import_module("simulation")

        run_ldpc_simulation = get_first_existing_attribute(
            ldpc_simulation,
            ["run_ldpc_simulation", "run_simulation"],
            "ldpc.simulation",
        )

        ldpc_ebn0_db = np.array(
            get_first_existing_attribute(
                ldpc_config,
                ["LDPC_EB_NO_DB", "EBN0_DECIBELS", "EB_NO_DECIBELS", "TURBO_LIKE_EBN0_DB"],
                "ldpc.config",
            ),
            dtype=float,
        )
        ldpc_iteration_list = list(
            get_first_existing_attribute(
                ldpc_config,
                ["DECODER_ITERATION_LIST", "ITERATION_LIST", "ITERATIONS"],
                "ldpc.config",
            )
        )

        start_time = time.perf_counter()
        simulation_output = run_ldpc_simulation()
        elapsed_seconds = time.perf_counter() - start_time

        if not isinstance(simulation_output, tuple) or len(simulation_output) < 2:
            raise ValueError("LDPC simulation must return at least (ber_results, llr_snapshot).")

        ldpc_results = simulation_output[0]
        ldpc_llr_snapshot = simulation_output[1]

        information_length = getattr(
            ldpc_config,
            "INFORMATION_BIT_COUNT",
            getattr(ldpc_config, "INFORMATION_BLOCK_LENGTH", None),
        )

        return {
            "rate": code_rate,
            "ldpc_ebn0_db": ldpc_ebn0_db,
            "iteration_list": ldpc_iteration_list,
            "ber_by_iteration": normalize_iteration_results(ldpc_results, ldpc_iteration_list),
            "llr_snapshot": ldpc_llr_snapshot,
            "information_length": information_length,
            "elapsed_seconds": elapsed_seconds,
        }


def benchmark_ldpc_iterations(base_rate: float = 1 / 3):
    """Measure LDPC runtime as the maximum decoder iteration count changes."""
    if not LDPC_FOLDER.exists():
        raise FileNotFoundError(f"LDPC folder not found: {LDPC_FOLDER}")

    measurements = {}

    for iteration_count in THROUGHPUT_ITERATION_LIST:
        with TemporaryProjectPath(LDPC_FOLDER):
            ldpc_config = importlib.import_module("config")
            apply_requested_rate(ldpc_config, base_rate)
            apply_requested_iteration_list(ldpc_config, list(range(1, iteration_count + 1)))

            ldpc_simulation = importlib.import_module("simulation")
            run_ldpc_simulation = get_first_existing_attribute(
                ldpc_simulation,
                ["run_ldpc_simulation", "run_simulation"],
                "ldpc.simulation",
            )

            start_time = time.perf_counter()
            run_ldpc_simulation()
            elapsed_seconds = time.perf_counter() - start_time
            measurements[iteration_count] = elapsed_seconds

    return measurements


# ---------------------------------------------------------------------
# Plot 1: BER vs SNR for several code rates
# ---------------------------------------------------------------------
def plot_ber_vs_snr_by_rate(turbo_results_by_rate, ldpc_results_by_rate):
    """
    Create a BER plot for each requested code rate.

    For each supported rate:
    - turbo best iteration curve,
    - LDPC best iteration curve,
    - uncoded BPSK reference.
    """
    supported_rates = sorted(
        set(turbo_results_by_rate.keys()).union(set(ldpc_results_by_rate.keys()))
    )

    if not supported_rates:
        print("No code rates could be plotted.")
        return

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11.2, 8.2),
        sharex=False,
        sharey=False,
    )
    axes = axes.ravel()
    floor_value = 1e-8

    for plot_index, code_rate in enumerate(CODE_RATES_TO_COMPARE):
        axis = axes[plot_index]
        axis.set_title(f"Code rate {format_rate_label(code_rate)}")

        turbo_data = turbo_results_by_rate.get(code_rate)
        ldpc_data = ldpc_results_by_rate.get(code_rate)

        if turbo_data is None and ldpc_data is None:
            axis.text(
                0.5,
                0.5,
                "Rate not supported\nby current repositories",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.grid(True, which="both", alpha=0.35)
            axis.set_xlabel("Eb/N0 (dB)")
            axis.set_ylabel("Bit error rate")
            continue

        reference_ebn0 = None
        if turbo_data is not None:
            reference_ebn0 = turbo_data["turbo_ebn0_db"]
        elif ldpc_data is not None:
            reference_ebn0 = ldpc_data["ldpc_ebn0_db"]

        axis.semilogy(
            reference_ebn0,
            np.clip(uncoded_bpsk_ber(reference_ebn0), floor_value, None),
            "r.-",
            linewidth=1.1,
            label="Uncoded BPSK",
        )

        if turbo_data is not None:
            best_turbo_iteration = max(turbo_data["iteration_list"])
            axis.semilogy(
                turbo_data["turbo_ebn0_db"],
                np.clip(
                    turbo_data["ber_by_iteration"][best_turbo_iteration],
                    floor_value,
                    None,
                ),
                "ko-",
                linewidth=1.3,
                label=f"Turbo, iteration {best_turbo_iteration}",
            )

        if ldpc_data is not None:
            best_ldpc_iteration = max(ldpc_data["iteration_list"])
            axis.semilogy(
                ldpc_data["ldpc_ebn0_db"],
                np.clip(
                    ldpc_data["ber_by_iteration"][best_ldpc_iteration],
                    floor_value,
                    None,
                ),
                color="tab:green",
                marker="D",
                linewidth=1.3,
                label=f"LDPC, iteration {best_ldpc_iteration}",
            )

        axis.grid(True, which="both", alpha=0.35)
        axis.set_xlabel("Eb/N0 (dB)")
        axis.set_ylabel("Bit error rate")
        axis.legend(fontsize=8)

    figure.suptitle("BER vs SNR comparison for Turbo and LDPC at several code rates")
    finish_figure(f"{FIGURE_PREFIX}_ber_by_rate.png")


# ---------------------------------------------------------------------
# Plot 2: Throughput and runtime versus iterations
# ---------------------------------------------------------------------
def plot_throughput_and_runtime(turbo_runtime_by_iteration, ldpc_runtime_by_iteration):
    """
    Plot decoding runtime and a simple relative throughput proxy.

    Throughput proxy:
        1 / runtime
    This is intentionally presented as a relative measure because the underlying
    simulations may use adaptive stopping and different frame counts.
    """
    iteration_counts = np.array(THROUGHPUT_ITERATION_LIST, dtype=int)
    turbo_runtime = np.array([turbo_runtime_by_iteration[it] for it in iteration_counts], dtype=float)
    ldpc_runtime = np.array([ldpc_runtime_by_iteration[it] for it in iteration_counts], dtype=float)

    turbo_relative_throughput = 1.0 / np.maximum(turbo_runtime, 1e-12)
    ldpc_relative_throughput = 1.0 / np.maximum(ldpc_runtime, 1e-12)

    plt.figure(figsize=(11.0, 4.8))

    plt.subplot(1, 2, 1)
    plt.plot(iteration_counts, turbo_runtime, "o-", label="Turbo runtime")
    plt.plot(iteration_counts, ldpc_runtime, "s--", label="LDPC runtime")
    plt.xlabel("Maximum decoder iteration count")
    plt.ylabel("Elapsed time (seconds)")
    plt.title("Runtime versus decoder iterations")
    plt.grid(True, alpha=0.35)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(iteration_counts, turbo_relative_throughput, "o-", label="Turbo throughput proxy")
    plt.plot(iteration_counts, ldpc_relative_throughput, "s--", label="LDPC throughput proxy")
    plt.xlabel("Maximum decoder iteration count")
    plt.ylabel("Relative throughput (1 / runtime)")
    plt.title("Relative decoding throughput versus iterations")
    plt.grid(True, alpha=0.35)
    plt.legend()

    finish_figure(f"{FIGURE_PREFIX}_throughput.png")


# ---------------------------------------------------------------------
# Printed summary
# ---------------------------------------------------------------------
def print_summary(turbo_results_by_rate, ldpc_results_by_rate, turbo_runtime_by_iteration, ldpc_runtime_by_iteration):
    """Print a compact text summary after generating the plots."""
    print("\nSummary of supported code rates")
    print("-" * 60)

    for code_rate in CODE_RATES_TO_COMPARE:
        turbo_supported = code_rate in turbo_results_by_rate
        ldpc_supported = code_rate in ldpc_results_by_rate
        print(
            f"Rate {format_rate_label(code_rate)}: "
            f"Turbo={'yes' if turbo_supported else 'no'}, "
            f"LDPC={'yes' if ldpc_supported else 'no'}"
        )

    print("\nRuntime by maximum iteration count")
    print("-" * 60)
    print(f"{'Iterations':>10} {'Turbo(s)':>12} {'LDPC(s)':>12}")
    for iteration_count in THROUGHPUT_ITERATION_LIST:
        print(
            f"{iteration_count:10d} "
            f"{turbo_runtime_by_iteration[iteration_count]:12.4f} "
            f"{ldpc_runtime_by_iteration[iteration_count]:12.4f}"
        )


# ---------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------
def main():
    turbo_results_by_rate = {}
    ldpc_results_by_rate = {}

    print("Running BER-versus-rate comparison...")
    for code_rate in CODE_RATES_TO_COMPARE:
        print(f"\nRequested code rate: {format_rate_label(code_rate)}")

        try:
            turbo_result = run_turbo_project_for_rate(code_rate)
            if turbo_result is not None:
                turbo_results_by_rate[code_rate] = turbo_result
        except Exception as exc:
            print(f"Turbo run failed for rate {format_rate_label(code_rate)}: {exc}")

        try:
            ldpc_result = run_ldpc_project_for_rate(code_rate)
            if ldpc_result is not None:
                ldpc_results_by_rate[code_rate] = ldpc_result
        except Exception as exc:
            print(f"LDPC run failed for rate {format_rate_label(code_rate)}: {exc}")

    print("\nRunning throughput / iteration benchmark...")
    turbo_runtime_by_iteration = benchmark_turbo_iterations()
    ldpc_runtime_by_iteration = benchmark_ldpc_iterations()

    plot_ber_vs_snr_by_rate(turbo_results_by_rate, ldpc_results_by_rate)
    plot_throughput_and_runtime(turbo_runtime_by_iteration, ldpc_runtime_by_iteration)
    print_summary(
        turbo_results_by_rate,
        ldpc_results_by_rate,
        turbo_runtime_by_iteration,
        ldpc_runtime_by_iteration,
    )


if __name__ == "__main__":
    main()

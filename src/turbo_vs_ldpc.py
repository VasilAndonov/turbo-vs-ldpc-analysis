from __future__ import annotations

"""
Comparison script for the turbo-code and LDPC-code projects.

This file runs both simulations and creates comparison plots that use the same
Eb/N0 axis and similar styling so the two coding schemes can be discussed side by side.

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
└── turbo_vs_ldpc.py
"""

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc


TURBO_REPOSITORY_FOLDER = Path("turbo")
LDPC_REPOSITORY_FOLDER = Path("ldpc")

OUTPUT_FOLDER = Path("comparison_results")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

ITERATIONS_TO_OVERLAY = [1, 3, 6]
SAVE_FIGURES = True
SHOW_FIGURES = True
FIGURE_PREFIX = "turbo_vs_ldpc"


COMMON_MODULE_NAMES = ["config", "encoder", "decoder", "simulation", "main"]


def clear_project_modules():
    """Remove repo-local module names so the next project can be imported cleanly."""
    for module_name in COMMON_MODULE_NAMES:
        sys.modules.pop(module_name, None)


class TemporaryImportPath:
    """Context manager that temporarily adds one project folder to Python's import path."""

    def __init__(self, folder: Path):
        self.folder = str(folder.resolve())

    def __enter__(self):
        sys.path.insert(0, self.folder)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if sys.path and sys.path[0] == self.folder:
            sys.path.pop(0)
        else:
            try:
                sys.path.remove(self.folder)
            except ValueError:
                pass
        clear_project_modules()


def get_first_existing_attribute(module, candidate_names, module_label):
    """Return the first attribute that exists from a list of possible names."""
    for name in candidate_names:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(
        f"Could not find any of these names in {module_label}: {candidate_names}"
    )


def build_iteration_dictionary(raw_results, iteration_list):
    """Normalize result containers into a dictionary keyed by iteration count."""
    if isinstance(raw_results, dict):
        return {int(iteration): np.array(raw_results[iteration], dtype=float) for iteration in iteration_list}

    if isinstance(raw_results, (list, tuple)):
        if len(raw_results) != len(iteration_list):
            raise ValueError(
                "Result container length does not match the iteration list length."
            )
        return {
            int(iteration_list[index]): np.array(raw_results[index], dtype=float)
            for index in range(len(iteration_list))
        }

    raise TypeError("Unsupported result container type for BER results.")


def run_turbo_project(turbo_folder: Path):
    if not turbo_folder.exists():
        raise FileNotFoundError(f"Turbo repository folder not found: {turbo_folder}")

    with TemporaryImportPath(turbo_folder):
        turbo_config = importlib.import_module("config")
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

        convolutional_ber = run_convolutional_simulation(random_generator)
        turbo_results, turbo_llr_snapshot = run_turbo_simulation(random_generator)

        turbo_data = {
            "convolutional_ebn0_db": convolutional_ebn0_db,
            "convolutional_ber": np.array(convolutional_ber, dtype=float),
            "turbo_ebn0_db": turbo_ebn0_db,
            "turbo_iteration_list": turbo_iteration_list,
            "turbo_ber_by_iteration": build_iteration_dictionary(
                turbo_results, turbo_iteration_list
            ),
            "turbo_llr_snapshot": turbo_llr_snapshot,
            "information_block_length": getattr(
                turbo_config, "INFORMATION_BLOCK_LENGTH", getattr(turbo_config, "K", None)
            ),
        }

    return turbo_data


def run_ldpc_project(ldpc_folder: Path):
    if not ldpc_folder.exists():
        raise FileNotFoundError(f"LDPC repository folder not found: {ldpc_folder}")

    with TemporaryImportPath(ldpc_folder):
        ldpc_config = importlib.import_module("config")
        ldpc_simulation = importlib.import_module("simulation")

        run_ldpc_simulation = get_first_existing_attribute(
            ldpc_simulation,
            ["run_ldpc_simulation", "run_simulation"],
            "ldpc.simulation",
        )

        ldpc_ebn0_db = np.array(
            get_first_existing_attribute(
                ldpc_config,
                ["EBN0_DECIBELS", "EB_NO_DECIBELS", "TURBO_LIKE_EBN0_DB"],
                "ldpc.config",
            ),
            dtype=float,
        )

        ldpc_iteration_list = list(
            get_first_existing_attribute(
                ldpc_config,
                ["ITERATION_LIST", "DECODER_ITERATION_LIST", "ITERATIONS"],
                "ldpc.config",
            )
        )

        simulation_output = run_ldpc_simulation()

        if not isinstance(simulation_output, tuple):
            raise TypeError(
                "LDPC simulation should return at least a tuple: (results, llr_snapshot)"
            )
        if len(simulation_output) < 2:
            raise ValueError(
                "LDPC simulation returned too few outputs. Expected (results, llr_snapshot)."
            )

        ldpc_results, ldpc_llr_snapshot = simulation_output[0], simulation_output[1]

        ldpc_data = {
            "ldpc_ebn0_db": ldpc_ebn0_db,
            "ldpc_iteration_list": ldpc_iteration_list,
            "ldpc_ber_by_iteration": build_iteration_dictionary(
                ldpc_results, ldpc_iteration_list
            ),
            "ldpc_llr_snapshot": ldpc_llr_snapshot,
            "information_block_length": getattr(
                ldpc_config,
                "INFORMATION_BIT_COUNT",
                getattr(ldpc_config, "INFORMATION_BLOCK_LENGTH", None),
            ),
            "code_rate": getattr(ldpc_config, "CODE_RATE", None),
        }

    return ldpc_data


def make_uncoded_bpsk_curve(ebn0_db_values: np.ndarray) -> np.ndarray:
    """Return the theoretical BER of uncoded BPSK in AWGN."""
    return 0.5 * erfc(np.sqrt(10.0 ** (np.asarray(ebn0_db_values, dtype=float) / 10.0)))


def save_or_show_figure(figure_name: str):
    output_path = OUTPUT_FOLDER / figure_name
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(output_path, dpi=170)
        print(f"Saved: {output_path}")
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close()


def plot_iteration_overlay(turbo_data: dict, ldpc_data: dict):
    floor_value = 1e-8
    plt.figure(figsize=(8.2, 5.2))

    for iteration_count in ITERATIONS_TO_OVERLAY:
        if iteration_count in turbo_data["turbo_ber_by_iteration"]:
            plt.semilogy(
                turbo_data["turbo_ebn0_db"],
                np.clip(turbo_data["turbo_ber_by_iteration"][iteration_count], floor_value, None),
                marker="o",
                linewidth=1.4,
                label=f"Turbo, iteration {iteration_count}",
            )
        if iteration_count in ldpc_data["ldpc_ber_by_iteration"]:
            plt.semilogy(
                ldpc_data["ldpc_ebn0_db"],
                np.clip(ldpc_data["ldpc_ber_by_iteration"][iteration_count], floor_value, None),
                marker="x",
                linewidth=1.4,
                linestyle="--",
                label=f"LDPC, iteration {iteration_count}",
            )

    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Bit error rate")
    plt.title("Turbo code vs LDPC code across equal iteration counts")
    plt.grid(True, which="both", alpha=0.35)
    plt.legend(ncol=2)
    save_or_show_figure(f"{FIGURE_PREFIX}_iterations.png")


def plot_best_curves_with_baselines(turbo_data: dict, ldpc_data: dict):
    floor_value = 1e-8
    best_turbo_iteration = max(turbo_data["turbo_iteration_list"])
    best_ldpc_iteration = max(ldpc_data["ldpc_iteration_list"])

    uncoded_curve = make_uncoded_bpsk_curve(turbo_data["convolutional_ebn0_db"])

    plt.figure(figsize=(8.2, 5.2))
    plt.semilogy(
        turbo_data["convolutional_ebn0_db"],
        np.clip(uncoded_curve, floor_value, None),
        "r.-",
        linewidth=1.2,
        label="Uncoded BPSK",
    )
    plt.semilogy(
        turbo_data["convolutional_ebn0_db"],
        np.clip(turbo_data["convolutional_ber"], floor_value, None),
        "bs-",
        linewidth=1.4,
        label="Convolutional baseline",
    )
    plt.semilogy(
        turbo_data["turbo_ebn0_db"],
        np.clip(turbo_data["turbo_ber_by_iteration"][best_turbo_iteration], floor_value, None),
        "ko-",
        linewidth=1.5,
        label=f"Turbo, iteration {best_turbo_iteration}",
    )
    plt.semilogy(
        ldpc_data["ldpc_ebn0_db"],
        np.clip(ldpc_data["ldpc_ber_by_iteration"][best_ldpc_iteration], floor_value, None),
        color="tab:green",
        marker="D",
        linewidth=1.5,
        label=f"LDPC, iteration {best_ldpc_iteration}",
    )

    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Bit error rate")
    plt.title("Best turbo and LDPC curves with reference baselines")
    plt.grid(True, which="both", alpha=0.35)
    plt.legend()
    save_or_show_figure(f"{FIGURE_PREFIX}_best_curves.png")


def plot_decoder_confidence_growth(turbo_data: dict, ldpc_data: dict):
    turbo_iterations = turbo_data["turbo_iteration_list"]
    ldpc_iterations = ldpc_data["ldpc_iteration_list"]

    plt.figure(figsize=(10.5, 4.6))

    plt.subplot(1, 2, 1)
    for iteration_count in turbo_iterations:
        sample = turbo_data["turbo_llr_snapshot"].get(iteration_count)
        if sample is not None:
            plt.scatter(range(len(sample)), sample, s=18, label=f"Iteration {iteration_count}")
    plt.title("Turbo decoder posterior LLR values")
    plt.xlabel("Bit index")
    plt.ylabel("Posterior LLR")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=8)

    plt.subplot(1, 2, 2)
    for iteration_count in ldpc_iterations:
        sample = ldpc_data["ldpc_llr_snapshot"].get(iteration_count)
        if sample is not None:
            plt.scatter(range(len(sample)), sample, s=18, label=f"Iteration {iteration_count}")
    plt.title("LDPC decoder posterior LLR values")
    plt.xlabel("Bit index")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=8)

    save_or_show_figure(f"{FIGURE_PREFIX}_llr_growth.png")


def print_quick_summary(turbo_data: dict, ldpc_data: dict):
    best_turbo_iteration = max(turbo_data["turbo_iteration_list"])
    best_ldpc_iteration = max(ldpc_data["ldpc_iteration_list"])

    common_ebn0_values = np.intersect1d(
        np.round(turbo_data["turbo_ebn0_db"], 6),
        np.round(ldpc_data["ldpc_ebn0_db"], 6),
    )

    print("\nQuick numerical summary")
    print("-" * 60)

    if turbo_data["information_block_length"] is not None:
        print(f"Turbo information block length: {turbo_data['information_block_length']}")
    if ldpc_data["information_block_length"] is not None:
        print(f"LDPC information block length: {ldpc_data['information_block_length']}")

    print(f"Compared turbo iteration: {best_turbo_iteration}")
    print(f"Compared LDPC iteration: {best_ldpc_iteration}")

    if common_ebn0_values.size > 0:
        print("\nBER at common Eb/N0 points using the highest configured iteration")
        print(f"{'Eb/N0 (dB)':>10} {'Turbo BER':>14} {'LDPC BER':>14} {'Lower BER':>12}")
        for ebn0_value in common_ebn0_values:
            turbo_index = np.where(np.isclose(turbo_data["turbo_ebn0_db"], ebn0_value))[0][0]
            ldpc_index = np.where(np.isclose(ldpc_data["ldpc_ebn0_db"], ebn0_value))[0][0]

            turbo_ber = turbo_data["turbo_ber_by_iteration"][best_turbo_iteration][turbo_index]
            ldpc_ber = ldpc_data["ldpc_ber_by_iteration"][best_ldpc_iteration][ldpc_index]
            winner = "Turbo" if turbo_ber < ldpc_ber else "LDPC"

            print(f"{ebn0_value:10.2f} {turbo_ber:14.4e} {ldpc_ber:14.4e} {winner:>12}")
    else:
        print("No common Eb/N0 samples were found between the two projects.")


def main():
    print("Running turbo-code simulation...")
    turbo_data = run_turbo_project(TURBO_REPOSITORY_FOLDER)

    print("\nRunning LDPC simulation...")
    ldpc_data = run_ldpc_project(LDPC_REPOSITORY_FOLDER)

    plot_iteration_overlay(turbo_data, ldpc_data)
    plot_best_curves_with_baselines(turbo_data, ldpc_data)
    plot_decoder_confidence_growth(turbo_data, ldpc_data)
    print_quick_summary(turbo_data, ldpc_data)


if __name__ == "__main__":
    main()

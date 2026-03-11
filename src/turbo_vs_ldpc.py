from __future__ import annotations

"""Comparison script for the turbo-code and LDPC-code projects.

This file runs both simulations and creates comparison plots that use the same
Eb/N0 axis and similar styling so the two coding schemes can be discussed side by side.

Expected folder layout
----------------------
.
├── compare_turbo_ldpc.py
├── turbo_repo/
│   ├── config.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── simulation.py
│   └── main.py
└── ldpc_repo/
    ├── config.py
    ├── encoder.py
    ├── decoder.py
    ├── simulation.py
    └── main.py
"""

import importlib
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
# These folder names are the only things you usually need to adjust.
TURBO_REPOSITORY_FOLDER = Path("turbo")
LDPC_REPOSITORY_FOLDER = Path("ldpc")

# The output folder is created automatically and stores only the comparison plots.
OUTPUT_FOLDER = Path("comparison_results")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Pick a few iteration counts to keep the combined plot readable.
ITERATIONS_TO_OVERLAY = [1, 3, 6]
SAVE_FIGURES = True
SHOW_FIGURES = True
FIGURE_PREFIX = "turbo_vs_ldpc"


# -----------------------------------------------------------------------------
# Utilities for safely importing the two separate repositories
# -----------------------------------------------------------------------------
# Both projects use the same module names such as config.py and simulation.py.
# Because of that, they cannot be imported in the normal way at the same time.
# The helper below temporarily inserts one project folder into sys.path, imports
# its modules, uses them, and then removes those module names before loading the
# second project.
COMMON_MODULE_NAMES = ["config", "encoder", "decoder", "simulation", "main"]


def clear_project_modules():
    """Remove repo-local module names so the second project can be imported cleanly."""
    for module_name in COMMON_MODULE_NAMES:
        sys.modules.pop(module_name, None)


class TemporaryImportPath:
    """Context manager that temporarily adds one folder to Python's import path."""

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


# -----------------------------------------------------------------------------
# Load and run the turbo-code project
# -----------------------------------------------------------------------------
# The turbo project already contains a convolutional baseline and an iterative
# turbo-code simulation. We collect both because the convolutional baseline is a
# useful reference point in the final comparison figures.

def run_turbo_project(turbo_folder: Path):
    if not turbo_folder.exists():
        raise FileNotFoundError(f"Turbo repository folder not found: {turbo_folder}")

    with TemporaryImportPath(turbo_folder):
        turbo_config = importlib.import_module("config")
        turbo_simulation = importlib.import_module("simulation")

        random_generator = np.random.default_rng(turbo_config.RANDOM_SEED)
        convolutional_ber = turbo_simulation.run_conv(random_generator)
        turbo_results, turbo_llr_snapshot = turbo_simulation.run_turbo(random_generator)

        turbo_data = {
            "convolutional_ebn0_db": np.array(turbo_simulation.CONV_EBN0_DB, dtype=float),
            "convolutional_ber": np.array(convolutional_ber, dtype=float),
            "turbo_ebn0_db": np.array(turbo_simulation.TURBO_EBN0_DB, dtype=float),
            "turbo_iteration_list": list(turbo_simulation.ITERATIONS),
            "turbo_ber_by_iteration": {
                iteration_count: np.array(turbo_results[iteration_count], dtype=float)
                for iteration_count in turbo_simulation.ITERATIONS
            },
            "turbo_llr_snapshot": turbo_llr_snapshot,
            "information_block_length": getattr(turbo_config, "INFORMATION_BLOCK_LENGTH", None),
        }

    return turbo_data


# -----------------------------------------------------------------------------
# Load and run the LDPC project
# -----------------------------------------------------------------------------
# The LDPC project returns BER curves for several decoder iterations and an LLR
# snapshot. These outputs are structured to mirror the turbo project closely.

def run_ldpc_project(ldpc_folder: Path):
    if not ldpc_folder.exists():
        raise FileNotFoundError(f"LDPC repository folder not found: {ldpc_folder}")

    with TemporaryImportPath(ldpc_folder):
        ldpc_config = importlib.import_module("config")
        ldpc_simulation = importlib.import_module("simulation")

        ldpc_results, ldpc_llr_snapshot = ldpc_simulation.run_ldpc_simulation()

        ldpc_data = {
            "ldpc_ebn0_db": np.array(ldpc_config.EBN0_DECIBELS, dtype=float),
            "ldpc_iteration_list": list(ldpc_config.ITERATION_LIST),
            "ldpc_ber_by_iteration": {
                iteration_count: np.array(ldpc_results[iteration_count], dtype=float)
                for iteration_count in ldpc_config.ITERATION_LIST
            },
            "ldpc_llr_snapshot": ldpc_llr_snapshot,
            "information_block_length": getattr(ldpc_config, "INFORMATION_BIT_COUNT", None),
            "code_rate": getattr(ldpc_config, "CODE_RATE", None),
        }

    return ldpc_data


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
# These helpers keep the figures consistent and easy to reuse in reports.

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


# -----------------------------------------------------------------------------
# Comparison plots
# -----------------------------------------------------------------------------
# The three plots below answer slightly different questions:
#   1) How do turbo and LDPC behave across the same iteration counts?
#   2) Which one performs better at the final iteration, alongside the baseline?
#   3) How quickly does decoder confidence grow inside one frame?

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

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)

    for iteration_count in turbo_iterations:
        sample = turbo_data["turbo_llr_snapshot"].get(iteration_count)
        if sample is not None:
            axes[0].scatter(range(len(sample)), sample, s=18, label=f"Iteration {iteration_count}")
    axes[0].set_title("Turbo decoder posterior LLR values")
    axes[0].set_xlabel("Bit index")
    axes[0].set_ylabel("Posterior LLR")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=8)

    for iteration_count in ldpc_iterations:
        sample = ldpc_data["ldpc_llr_snapshot"].get(iteration_count)
        if sample is not None:
            axes[1].scatter(range(len(sample)), sample, s=18, label=f"Iteration {iteration_count}")
    axes[1].set_title("LDPC decoder posterior LLR values")
    axes[1].set_xlabel("Bit index")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=8)

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
        print(f"LDPC information block length:  {ldpc_data['information_block_length']}")
    print(f"Compared turbo iteration: {best_turbo_iteration}")
    print(f"Compared LDPC iteration:  {best_ldpc_iteration}")

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


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
# Running this file will execute both projects, create the comparison plots, and
# print a compact numerical summary that can be copied into a report.

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

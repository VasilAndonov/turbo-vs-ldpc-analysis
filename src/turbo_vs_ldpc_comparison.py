from __future__ import annotations
import importlib
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

TURBO_FOLDER = Path("turbo")
LDPC_FOLDER = Path("ldpc")
OUTPUT_FOLDER = Path("comparison_results")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

CODE_RATE_LABELS = ["1/2", "1/3", "3/4", "7/8"]
THROUGHPUT_REFERENCE_RATE = "1/3"

SAVE_FIGURES = True
SHOW_FIGURES = True
FIGURE_PREFIX = "turbo_vs_ldpc_rates"

PROJECT_MODULE_NAMES = ["config", "encoder", "decoder", "simulation", "main"]


def clear_project_modules():
    for name in PROJECT_MODULE_NAMES:
        sys.modules.pop(name, None)


class TemporaryProjectPath:
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


def patch_config_text(config_text: str, rate_label: str) -> str:
    return re.sub(
        r'SELECTED_CODE_RATE\s*=\s*SUPPORTED_CODE_RATES\["[^"]+"\]',
        f'SELECTED_CODE_RATE = SUPPORTED_CODE_RATES["{rate_label}"]',
        config_text,
    )


class TemporaryConfigPatch:
    def __init__(self, project_folder: Path, rate_label: str):
        self.project_folder = project_folder
        self.rate_label = rate_label
        self.config_path = project_folder / "config.py"
        self.original_text = None

    def __enter__(self):
        self.original_text = self.config_path.read_text()
        self.config_path.write_text(patch_config_text(self.original_text, self.rate_label))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.original_text is not None:
            self.config_path.write_text(self.original_text)
        clear_project_modules()


def uncoded_bpsk_ber(ebn0_db_values):
    ebn0_db_values = np.asarray(ebn0_db_values, dtype=float)
    return 0.5 * erfc(np.sqrt(10.0 ** (ebn0_db_values / 10.0)))


def finish_figure(file_name: str):
    output_path = OUTPUT_FOLDER / file_name
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(output_path, dpi=180)
        print(f"Saved: {output_path}")
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close()


def run_turbo_project_for_rate(rate_label: str):
    with TemporaryConfigPatch(TURBO_FOLDER, rate_label):
        with TemporaryProjectPath(TURBO_FOLDER):
            turbo_config = importlib.import_module("config")
            turbo_simulation = importlib.import_module("simulation")
            rng = np.random.default_rng(getattr(turbo_config, "RANDOM_SEED", 12))
            convolutional_ber = turbo_simulation.run_convolutional_simulation(rng)
            turbo_results, _ = turbo_simulation.run_turbo_simulation(rng)
            return {
                "convolutional_ebn0_db": np.array(turbo_config.CONVOLUTIONAL_EB_NO_DB, dtype=float),
                "convolutional_ber": np.array(convolutional_ber, dtype=float),
                "ebn0_db": np.array(turbo_config.TURBO_EB_NO_DB, dtype=float),
                "iteration_list": list(turbo_config.DECODER_ITERATION_LIST),
                "ber_by_iteration": {int(k): np.array(v, dtype=float) for k, v in turbo_results.items()},
            }


def run_ldpc_project_for_rate(rate_label: str):
    with TemporaryConfigPatch(LDPC_FOLDER, rate_label):
        with TemporaryProjectPath(LDPC_FOLDER):
            ldpc_config = importlib.import_module("config")
            ldpc_simulation = importlib.import_module("simulation")
            ldpc_results, _ = ldpc_simulation.run_ldpc_simulation()
            return {
                "ebn0_db": np.array(ldpc_config.LDPC_EB_NO_DB, dtype=float),
                "iteration_list": list(ldpc_config.DECODER_ITERATION_LIST),
                "ber_by_iteration": {int(k): np.array(v, dtype=float) for k, v in ldpc_results.items()},
            }


def benchmark_turbo_decoder(rate_label: str):
    with TemporaryConfigPatch(TURBO_FOLDER, rate_label):
        with TemporaryProjectPath(TURBO_FOLDER):
            turbo_config = importlib.import_module("config")
            turbo_simulation = importlib.import_module("simulation")
            rng = np.random.default_rng(getattr(turbo_config, "RANDOM_SEED", 12))
            return turbo_simulation.benchmark_turbo_decoder(rng)


def benchmark_ldpc_decoder(rate_label: str):
    with TemporaryConfigPatch(LDPC_FOLDER, rate_label):
        with TemporaryProjectPath(LDPC_FOLDER):
            ldpc_config = importlib.import_module("config")
            ldpc_simulation = importlib.import_module("simulation")
            rng = np.random.default_rng(getattr(ldpc_config, "RANDOM_SEED", 12))
            return ldpc_simulation.benchmark_ldpc_decoder(rng)


def plot_ber_vs_snr_by_rate(turbo_results_by_rate, ldpc_results_by_rate):
    floor_value = 1e-8
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    axes = axes.ravel()

    for plot_index, rate_label in enumerate(CODE_RATE_LABELS):
        axis = axes[plot_index]
        axis.set_title(f"Code rate {rate_label}")
        turbo_data = turbo_results_by_rate.get(rate_label)
        ldpc_data = ldpc_results_by_rate.get(rate_label)

        if turbo_data is None or ldpc_data is None:
            axis.text(
                0.5,
                0.5,
                "Rate could not be simulated\nfor one or both projects",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.grid(True, which="both", alpha=0.35)
            axis.set_xlabel("Eb/N0 (dB)")
            axis.set_ylabel("Bit error rate")
            continue

        reference_ebn0 = turbo_data["ebn0_db"]
        axis.semilogy(
            reference_ebn0,
            np.clip(uncoded_bpsk_ber(reference_ebn0), floor_value, None),
            "r.-",
            linewidth=1.1,
            label="Uncoded BPSK",
        )

        best_turbo_iteration = max(turbo_data["iteration_list"])
        axis.semilogy(
            turbo_data["ebn0_db"],
            np.clip(turbo_data["ber_by_iteration"][best_turbo_iteration], floor_value, None),
            "ko-",
            linewidth=1.3,
            label=f"Turbo, iteration {best_turbo_iteration}",
        )

        best_ldpc_iteration = max(ldpc_data["iteration_list"])
        axis.semilogy(
            ldpc_data["ebn0_db"],
            np.clip(ldpc_data["ber_by_iteration"][best_ldpc_iteration], floor_value, None),
            color="tab:green",
            marker="D",
            linewidth=1.3,
            label=f"LDPC, iteration {best_ldpc_iteration}",
        )

        axis.grid(True, which="both", alpha=0.35)
        axis.set_xlabel("Eb/N0 (dB)")
        axis.set_ylabel("Bit error rate")
        axis.legend(fontsize=8)

    figure.suptitle("BER versus SNR for Turbo and LDPC at several code rates")
    finish_figure(f"{FIGURE_PREFIX}_ber_by_rate.png")


def plot_throughput_and_runtime(turbo_runtime_by_iteration, ldpc_runtime_by_iteration):
    iteration_counts = np.array(sorted(turbo_runtime_by_iteration.keys()), dtype=int)
    turbo_runtime = np.array([turbo_runtime_by_iteration[count] for count in iteration_counts], dtype=float)
    ldpc_runtime = np.array([ldpc_runtime_by_iteration[count] for count in iteration_counts], dtype=float)

    turbo_relative_throughput = 1.0 / np.maximum(turbo_runtime, 1e-12)
    ldpc_relative_throughput = 1.0 / np.maximum(ldpc_runtime, 1e-12)

    plt.figure(figsize=(11.0, 4.8))
    plt.subplot(1, 2, 1)
    plt.plot(iteration_counts, turbo_runtime, "o-", label="Turbo runtime")
    plt.plot(iteration_counts, ldpc_runtime, "s--", label="LDPC runtime")
    plt.xlabel("Maximum decoder iteration count")
    plt.ylabel("Elapsed time (seconds)")
    plt.title(f"Decoder runtime at rate {THROUGHPUT_REFERENCE_RATE}")
    plt.grid(True, alpha=0.35)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(iteration_counts, turbo_relative_throughput, "o-", label="Turbo throughput proxy")
    plt.plot(iteration_counts, ldpc_relative_throughput, "s--", label="LDPC throughput proxy")
    plt.xlabel("Maximum decoder iteration count")
    plt.ylabel("Relative throughput (1 / runtime)")
    plt.title(f"Relative decoder throughput at rate {THROUGHPUT_REFERENCE_RATE}")
    plt.grid(True, alpha=0.35)
    plt.legend()

    finish_figure(f"{FIGURE_PREFIX}_throughput.png")


def main():
    turbo_results_by_rate = {}
    ldpc_results_by_rate = {}

    print("Running BER-versus-rate comparison...")
    for rate_label in CODE_RATE_LABELS:
        print(f"Running rate {rate_label}...")
        try:
            turbo_results_by_rate[rate_label] = run_turbo_project_for_rate(rate_label)
        except Exception as exc:
            print(f"Turbo run failed for rate {rate_label}: {exc}")

        try:
            ldpc_results_by_rate[rate_label] = run_ldpc_project_for_rate(rate_label)
        except Exception as exc:
            print(f"LDPC run failed for rate {rate_label}: {exc}")

    print("Running fixed-workload decoder benchmark...")
    turbo_runtime_by_iteration = benchmark_turbo_decoder("1/3")
    ldpc_runtime_by_iteration = benchmark_ldpc_decoder("1/3")

    plot_ber_vs_snr_by_rate(turbo_results_by_rate, ldpc_results_by_rate)
    plot_throughput_and_runtime(turbo_runtime_by_iteration, ldpc_runtime_by_iteration)


if __name__ == "__main__":
    main()

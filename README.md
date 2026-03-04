# Comparative Analysis of Turbo and LDPC Codes

## Project Overview

This project presents a scientific and experimental comparative analysis of **Turbo Codes** and **Low-Density Parity-Check (LDPC) Codes**, two advanced forward error correction (FEC) techniques widely used in modern communication systems.

The work integrates mathematical modeling, algorithmic implementation, and simulation-based validation to evaluate performance, decoding complexity, similar behavior, and practical trade-offs over an Additive White Gaussian Noise (AWGN) channel.

---

## Research Objectives

The primary objectives of this project are:

- Evaluate and compare Bit Error Rate (BER) performance  
- Analyze iterative decoding convergence behavior  
- Compare computational complexity and decoding time  
- Test the influence of code rate and redundancy  
- Clarify the distinction between error detection and error correction  
- Determine practical criteria for selecting Turbo or LDPC codes  

---

## Methodology

The comparison is conducted under simulation conditions:

- **Channel Model:** Additive White Gaussian Noise (AWGN)  
- **Modulation Scheme:** Binary Phase Shift Keying (BPSK)  
- **Performance Metric:** Bit Error Rate (BER)  
- Identical SNR range for both coding schemes  
- Fixed maximum number of decoding iterations  
- Comparable block lengths and code rates  

Both coding schemes are implemented in Python, and experiments are conducted with identical parameters to ensure a fair and objective comparison.

---

## Repository Structure

- `notebook/` – Main analysis with theory, implementation, and experiments  
- `src/` – Implementation of Turbo, LDPC, and channel models
- `experiments/` – Saved simulation results for reproducibility  
- `figures/` – Generated plots for BER, iterations, and decoding time
- `docs/` – Contains additional documentation with all cited sources and bibliographic information  

---

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```
2. Launch Jupyter Notebook:

```bash
jupyter notebook
```
3. Open and run the main notebook:

```bash
notebook/comparative-analysis-turbo-ldpc.ipynb
```

## Tools

- Python
- NumPy
- SciPy
- Matplotlib
- Pandas
- Jupyter

---

## Academic Context

This project is developed as part of a math course. It follows a scientific methodology including problem formulation, mathematical modeling, algorithm implementation, experimental validation, and comparative analysis.

---

## References

Foundational references for Turbo Codes, LDPC Codes, Shannon capacity theory, and modern communication standards are provided within the notebook.

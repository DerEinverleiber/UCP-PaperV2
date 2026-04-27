# Scaling Quantum Simulation-Based Optimization
### Demonstrating Efficient Power Grid Management with Deep QAOA Circuits

Code accompanying the paper *Scaling Quantum Simulation-Based Optimization: Demonstrating Efficient Power Grid Management with Deep QAOA Circuits*, submitted to IEEE QCE 2026. This repository benchmarks the optimization component of a QAOA-based Quantum Simulation-based Optimization (QuSO) solver against simulated annealing on randomly generated power grid instances.

| Avg. RAAR | Max qubits | Max initial QAOA layers |
|-----------|------------|--------------------------|
| 69%       | 14         | 128                      |

---

## Installation

```bash
pip install pennylane numpy scipy matplotlib
```

---

## Quickstart

To reproduce the main benchmark results, first generate power grid instances and precompute cost values, then run the QAOA parameter training, and finally collect results:

```bash
python qaoa_calc_parameters.py
python qaoa_calc_results.py
```

Plots used in the paper can be reproduced by running all cells in `qaoa_plots.ipynb` and `cl_optimizer_plots.ipynb`.

---

## Repository Structure

### Core modules

| File | Description |
|------|-------------|
| `qaoa_pipeline.py` | QAOA circuit construction and iterative interpolation parameter training [(Apte et al.)](https://arxiv.org/abs/2504.01694) |
| `powergrid.py` | Random power grid generation (DC approximation), used as benchmark instances |
| `metrics.py` | RAAR and TTS* implementations for QAOA and simulated annealing |
| `brute_force.py` | Exhaustive cost function evaluation over the full solution space |
| `qaoa_calc_parameters.py` | Trains QAOA angles via iterative interpolation |
| `qaoa_calc_results.py` | Computes RAAR and TTS* from trained parameters |

### Notebooks

| File | Description |
|------|-------------|
| `qaoa_plots.ipynb` | Generates all QAOA figures used in the paper |
| `cl_optimizer_plots.ipynb` | Generates all simulated annealing figures used in the paper |

<details>
<summary>Legacy notebooks (potentially outdated)</summary>

| File | Description |
|------|-------------|
| `test_powergrid.ipynb` | Justifies using randomly generated power grids rather than the IEEE 57-bus dataset. Also contains testing of the cost function. |
| `test_qaoa_maxcut.ipynb` | Sanity-checks `qaoa_pipeline.py` on the MaxCut problem |

</details>

---


## Authors

Jonas Stein · Jannis Lutz ([@DerEinverleiber](https://github.com/DerEinverleiber)) · Moritz Solderer ([@moritzsoelderer](https://github.com/moritzsoelderer)) · Maximilian Adler · Michael Lachner · David Bucher · Claudia Linnhoff-Popien

Correspondence: [jonas.stein@ifi.lmu.de](mailto:jonas.stein@ifi.lmu.de)

---

## Acknowledgements

Supported by the LMU Sustainability Fund (EfOiE), BMFTR (QuCUN, QuaRDS, CAQAO), Munich Quantum Valley (K5, K7), and Bavarian StMWi (6GQT).
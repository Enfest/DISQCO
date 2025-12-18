# Quantum Circuit Partitioning Benchmarking

This directory contains benchmarking suites for two research papers on quantum circuit partitioning.

## Organization

The benchmarking code is organized by paper:

### **MFPQC/** - Multilevel Framework for Partitioning Quantum Circuits
Circuit partitioning over all-to-all networks with the multilevel Fiduccia-Mattheyses.

**Key Features:**
- CP fraction circuits with scaling analysis
- Standard quantum circuits (QAOA, QFT, QV)
- QASMBench real-world circuits
- Coarsener comparison studies
- All-to-all network topology

**See:** `MFPQC/README.md` for details

---

### **EEDQC/** - Entanglement-Efficient Distribution of Quantum Circuits over Large-Scale Quantum Networks
General network topologies. Network coarsening and hierarchical recursive partitioning.

**Key Features:**
- Constrained network topologies (linear, grid, random)
- Net-coarsened partitioning for large networks
- CP and QASM circuits on heterogeneous networks
- Network coarsening factors

**See:** `EEDQC/README.md` for details

---

## Quick Start

### MFPQC Benchmarks
```bash
cd MFPQC
# Edit benchmark_config.yaml to configure
python run_all_benchmarks.py
```

### EEDQC Benchmarks
```bash
cd EEDQC
# Edit benchmark_config.yaml to configure
python run_all_benchmarks.py
```

## Common Features

Both suites include:
- **YAML Configuration**: Easy parameter management
- **Multiprocessing Support**: Parallel iteration execution
- **Progress Tracking**: Real-time tqdm progress bars
- **Incremental Saving**: Results saved after each iteration
- **JSON Output**: Detailed and aggregated result files
- **Analysis Tools**: Scripts for post-processing and visualization

## Output Structure

Each paper directory maintains its own:
- `data/`: Raw JSON benchmark results
- `analysis/`: Analysis scripts and processed data
- `scripts/`: Individual benchmark modules


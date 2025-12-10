# MLFM-R Benchmarking Suite

This directory contains benchmarking infrastructure for the Multilevel Fiduccia-Mattheyses Recursive (MLFM-R) quantum circuit partitioning algorithm.

## Directory Structure

```
benchmarking/
├── README.md                    # This file
├── benchmark_config.yaml        # Configuration for all benchmarks
├── run_all_benchmarks.py       # Main script to run all benchmarks
├── scripts/                     # Individual benchmark modules
│   ├── benchmark_cp_large.py          # Large CP circuits (112-240 qubits)
│   ├── benchmark_cp_scaling.py        # Scaling CP circuits (2-12 partitions)
│   ├── benchmark_qaoa.py              # QAOA circuits
│   ├── benchmark_qft.py               # QFT circuits
│   ├── benchmark_qv.py                # Quantum Volume circuits
│   ├── benchmark_qasm.py              # QASMBench circuits
│   ├── benchmark_cp_circuits.py       # Legacy wrapper
│   └── benchmark_standard_circuits.py # Legacy wrapper
├── analysis/                    # Analysis and plotting scripts
│   └── plot_results.py
├── data/                        # Raw benchmark results (not in git)
│   └── *.json
└── results/                     # Plots and summaries (committed to git)
    ├── plots/
    └── *.md
```

## Quick Start

### 1. Configure Benchmarks

Edit `benchmark_config.yaml` to set:
- Which benchmarks to run (`enabled: true/false`)
- Circuit sizes and partition counts
- Number of iterations per configuration
- Number of parallel processes (`num_processes`) for faster execution
- Algorithm parameters

**Performance Tip:** Set `num_processes` to utilize multiple CPU cores:
```yaml
num_iterations: 5
num_processes: 4  # Use 4 parallel processes (set to 1 to disable multiprocessing)
```

### 2. Run Benchmarks

Run all enabled benchmarks:

```bash
cd benchmarking
python run_all_benchmarks.py
```

This will:
- Run all enabled benchmarks from the config
- Display progress bars for each benchmark and iteration (via tqdm)
- Parallelize iterations across multiple CPU cores (if `num_processes > 1`)
- Save raw results to `data/*.json`
- Print progress and summary statistics
- Save metadata including git commit hash and timestamp

**Note:** Benchmarks can take several hours depending on configuration.

### 3. Generate Plots and Analysis

After benchmarks complete, generate plots:

```bash
cd analysis
python plot_results.py
```

This creates:
- `results/plots/*.png` - Publication-quality plots
- `results/*.md` - Summary tables in markdown format

## Benchmark Types

### CP Circuit Benchmarks - Large Scale
- Tests 2 and 4 partition configurations
- Circuit sizes: 112-240 qubits (configurable)
- Fixed fraction: 0.5
- Good for testing scalability

### CP Circuit Benchmarks - Scaling Partitions
- Tests increasing partition counts (2-12)
- Circuit sizes: 16-96 qubits
- Multiple fractions: 0.3, 0.5, 0.7, 0.9
- Good for understanding partition scaling

### Standard Circuit Benchmarks
- Tests QAOA, QFT, and Quantum Volume circuits
- 2 and 4 partition configurations
- Circuit sizes: 16-96 qubits
- Good for comparing across circuit types

### QASMBench Circuit Benchmarks
- Tests real-world circuits from QASMBench suite
- Configurable category (small, medium, large)
- Multiple partition configurations (2, 3, 4)
- Filters circuits by max depth
- **Requires QASMBench installation** (see below)

## Configuration File

`benchmark_config.yaml` structure:

```yaml
num_iterations: 5  # Repetitions per configuration
num_processes: 4   # Number of parallel processes (1 = serial execution)

cp_large:
  enabled: true
  sizes: [112, 128, 144, ...]
  num_partitions: [2, 4]
  fraction: 0.5

standard_circuits:
  enabled: true
  circuit_types: ['QAOA', 'QFT', 'QV']
  sizes: [16, 24, 32, ...]
  num_partitions: [2, 4]

qasm_circuits:
  enabled: false
  qasmbench_path: "QASMBench"
  category: "large"
  num_partitions: [2, 3, 4]

algorithm:
  move_limit_factor: 0.125
  basis_gates: ['u', 'cp']
```

## Performance Features

### Progress Tracking
All benchmark scripts use **tqdm** to display:
- Overall progress across configurations
- Per-configuration iteration progress
- Estimated time remaining

### Parallel Execution
Set `num_processes > 1` in `benchmark_config.yaml` to enable multiprocessing:
- Each configuration's iterations run in parallel
- Utilizes multiple CPU cores efficiently
- Typically 2-4x speedup with 4 processes
- Set to 1 for serial execution (useful for debugging)

**Example:** With 5 iterations and 4 processes, all 5 iterations run simultaneously instead of sequentially.

**Note:** Memory usage scales with `num_processes`. Reduce if you encounter memory issues.

## Output Files

### Raw Data (data/)
- `benchmark_results_*.json` - All individual iteration results
- `benchmark_means_*.json` - Aggregated mean results
- `metadata_*.json` - Run metadata (timestamp, git commit, config)

**Not committed to git** - Excluded via `.gitignore`

### Results (results/)
- `plots/*.png` - Generated plots
- `*_summary.md` - Summary statistics tables

**Committed to git** - For publication and sharing

## Reproducing Results

To reproduce published results:

1. Check out the corresponding git commit
2. Ensure you have the same environment (see `metadata_*.json`)
3. Run benchmarks with the same configuration
4. Compare results (some variation expected due to stochastic nature)

## Environment Setup

Required packages:
- qiskit
- numpy
- matplotlib
- pandas
- pyyaml
- tqdm (for progress bars)
- disqco (this package)

Install with:
```bash
pip install qiskit numpy matplotlib pandas pyyaml tqdm
pip install -e .  # Install disqco in development mode
```

### Optional: QASMBench Setup

For QASMBench benchmarks:

1. Clone QASMBench repository:
```bash
git clone https://github.com/pnnl/QASMBench.git
```

2. Update `benchmark_config.yaml` with the correct path:
```yaml
qasm_circuits:
  enabled: true
  qasmbench_path: "/path/to/QASMBench"
```

3. Ensure the QASMBench Python interface is available:
```bash
cd QASMBench
pip install -e .
```

## Adding New Benchmarks

1. Create a new module in `scripts/` (e.g., `benchmark_new_circuit.py`)
2. Implement a function that takes `(config, output_dir)` as parameters
3. Add configuration section to `benchmark_config.yaml`
4. Import and call your function in `run_all_benchmarks.py`
5. Update plotting script to handle new data format

## Performance Notes

Approximate runtime for default configuration:
- CP Large Scale: ~2-4 hours
- CP Scaling: ~3-5 hours  
- Standard Circuits: ~1-2 hours per circuit type

Total: ~10-15 hours for full benchmark suite

Consider:
- Running overnight or on a compute cluster
- Disabling benchmarks you don't need
- Reducing iteration count for quick tests
- Running circuit types separately

## Citation

If you use these benchmarks in your research, please cite:

```
[Your paper citation here]
```

## Questions?

For issues or questions about benchmarking:
- Check existing notebooks in `benchmarking/*.ipynb`
- See demos in `../demos/` for algorithm usage examples
- Open an issue on GitHub

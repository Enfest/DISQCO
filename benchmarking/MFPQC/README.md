# MFPQC Benchmarking Suite
**A Multilevel Framework for Partitioning Quantum Circuits**

This directory contains benchmarking scripts for the MFPQC paper, focusing on mutilevel partitioning with temporal coarsening. Most results use the MLFM-R algorithm (Multilevel Fiduccia-Mattheyses Recursive).

## Directory Structure

```
MFPQC/
├── scripts/              # Benchmark modules
│   ├── benchmark_cp_large.py
│   ├── benchmark_cp_scaling.py
│   ├── benchmark_qaoa.py
│   ├── benchmark_qft.py
│   ├── benchmark_qv.py
│   ├── benchmark_qasm_50.py
│   ├── benchmark_qasm_100.py
│   └── benchmark_coarsener_comparison.py
├── data/                 # Output JSON files
├── analysis/             # Analysis scripts and .dat files
│   ├── calculate_ebit_fractions.py
│   └── generate_dat_files.py
├── benchmark_config.yaml # Configuration file
└── run_all_benchmarks.py # Main runner script
```

## Running Benchmarks

1. **Configure**: Edit `benchmark_config.yaml` to enable/disable benchmarks and set parameters

2. **Run all benchmarks**:
   ```bash
   cd MFPQC
   python run_all_benchmarks.py
   ```

3. **Run individual benchmarks**:
   ```bash
   python scripts/benchmark_cp_scaling.py --config benchmark_config.yaml
   ```

## Benchmark Types

- **CP Scaling**: CP fraction circuits with 2-12 partitions
- **CP Large**: Large-scale CP circuits (112-256 qubits) with 2-4 partitions
- **Standard Circuits**: QAOA, QFT, QV benchmarks
- **QASMBench**: Real-world quantum circuits
- **Coarsener Comparison**: Compare different coarsening strategies

## Performance Features

- **Multiprocessing**: Parallel execution of iterations (`num_processes` in config)
- **Progress Tracking**: tqdm progress bars for real-time monitoring
- **Incremental Saving**: Results saved after each iteration

## Analysis Tools

- **E-bit Fractions**: Calculate normalized cost/time metrics
  ```bash
  python analysis/calculate_ebit_fractions.py --input_dir data
  ```

- **DAT File Generation**: Generate .dat files for plotting
  ```bash
  python analysis/generate_dat_files.py --input_dir data --output_dir analysis/dat_files
  ```

## Output Files

- `benchmark_results_*.json`: Detailed per-iteration results
- `benchmark_means_*.json`: Aggregated statistics
- `analysis/dat_files/*.dat`: Formatted data for plotting

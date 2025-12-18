# EEDQC Benchmarking Suite
**Entanglement-Efficient Distribution of Quantum Circuits over Large-Scale Quantum Networks**

This directory contains benchmarking scripts for the EEDQC paper, focusing on partitioning circuits over constrained quantum networks with various network topologies (linear, grid, random). Network coarsening and hierarchical recursive partitioning is used for large networks in ```benchmark_net_coarsened.py```.

## Directory Structure

```
EEDQC/
├── scripts/              # Benchmark modules
│   ├── benchmark_cp_hetero.py
│   ├── benchmark_qasm_hetero.py
│   └── benchmark_net_coarsened.py
├── data/                 # Output JSON files
├── analysis/             # Analysis scripts
├── benchmark_config.yaml # Configuration file
└── run_all_benchmarks.py # Main runner script
```

## Running Benchmarks

1. **Configure**: Edit `benchmark_config.yaml` to enable/disable benchmarks and set parameters

2. **Run all benchmarks**:
   ```bash
   cd EEDQC
   python run_all_benchmarks.py
   ```

3. **Run individual benchmarks**:
   ```bash
   python scripts/benchmark_cp_hetero.py --config benchmark_config.yaml
   ```

## Benchmark Types

- **CP Heterogeneous**: CP fraction circuits on heterogeneous networks
  - Network types: linear, grid, random
  - Sizes: 112-256 qubits
  - Partitions: 4, 6, 8

- **QASM Heterogeneous**: QASMBench circuits on heterogeneous networks
  - Real-world quantum algorithms
  - Multiple network topologies
  - Configurable partition counts

- **Net-Coarsened Partitioning**: Network coarsening for large-scale networks
  - Multiple coarsening factors (2, 4)
  - Large network sizes (8, 16, 32 QPUs)
  - Various circuit types (CP, QFT, QV)

## Key Features

### Heterogeneous Networks
- **Linear**: Chain topology for distributed systems
- **Grid**: 2D grid topology for scalability
- **Random**: Random connectivity for robust comparison

### Network Coarsening
Enables efficient partitioning of large networks by:
- Coarsening the network topology
- Recursive multilevel refinement
- Parallel execution support

## Performance Features

- **Multiprocessing**: Parallel execution of iterations
- **Progress Tracking**: Real-time monitoring with tqdm
- **Error Handling**: Graceful failure recovery for QASM circuits
- **Incremental Saving**: Results saved after each configuration

## Output Files

- `benchmark_results_EEDQC_*.json`: Detailed per-iteration results
- `benchmark_means_EEDQC_*.json`: Aggregated statistics with standard deviations
- Network-specific metrics for topology comparison

## Configuration

Key parameters in `benchmark_config.yaml`:
- `network_types`: List of network topologies to test
- `num_qpus`: Number of QPUs in the network
- `coarsening_factors`: Network coarsening factors
- `use_multiprocessing`: Enable parallel FM within net-coarsened partitioning

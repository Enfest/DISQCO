"""
Standard Circuit Benchmarking Module (Legacy - Use individual modules)

This module imports and re-exports individual circuit benchmarking functions.
Use benchmark_qaoa.py, benchmark_qft.py, and benchmark_qv.py directly for new code.
"""

from .benchmark_qaoa import run_qaoa_benchmark
from .benchmark_qft import run_qft_benchmark
from .benchmark_qv import run_qv_benchmark

__all__ = ['run_qaoa_benchmark', 'run_qft_benchmark', 'run_qv_benchmark', 'run_standard_circuit_benchmarks']


def run_standard_circuit_benchmarks(config, output_dir):
    """
    Run benchmarks for all standard circuits (QAOA, QFT, QV).
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save results
    """
    circuit_types = config['standard_circuits'].get('circuit_types', ['QAOA', 'QFT', 'QV'])
    
    for circuit_type in circuit_types:
        if circuit_type == "QAOA":
            run_qaoa_benchmark(config, output_dir)
        elif circuit_type == "QFT":
            run_qft_benchmark(config, output_dir)
        elif circuit_type == "QV":
            run_qv_benchmark(config, output_dir)
        else:
            print(f"Warning: Unknown circuit type '{circuit_type}' - skipping")

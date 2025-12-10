"""
CP Circuit Benchmarking Module (Legacy - Use individual modules)

This module imports and re-exports individual CP benchmarking functions.
Use benchmark_cp_large.py and benchmark_cp_scaling.py directly for new code.
"""

from .benchmark_cp_large import run_cp_large_benchmark
from .benchmark_cp_scaling import run_cp_scaling_benchmark

__all__ = ['run_cp_large_benchmark', 'run_cp_scaling_benchmark']

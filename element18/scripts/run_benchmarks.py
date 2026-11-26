"""Run benchmarks. Writes JSON to results/. Record hardware state yourself (power mode, temperature)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from element18 import benchmark

if __name__ == "__main__":
    print(benchmark.run_preprocessing())

"""
LADDER: Integration Problem Generator and Dataset Builder.

A Python package for generating mathematical integration problems and building datasets
for machine learning research.
"""

__version__ = "0.1.0"
__author__ = "Toby Simonds"
__email__ = "toby@example.com"

from ladder.generate_variants import process_integral
from ladder.batch_generator import BatchGenerator

__all__ = [
    "__version__",
    "process_integral",
    "BatchGenerator",
]

# LADDER 🪜

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**LADDER** is a modern Python package for generating mathematical integration problem variants and building datasets for machine learning research. Based on the [LADDER paper](https://arxiv.org/abs/2503.00735).

## ✨ Features

- 🧮 **Intelligent Variant Generation**: Create easier, equivalent, and harder variants of integration problems
- 🚀 **Async Processing**: High-performance batch processing with concurrent execution  
- 🔧 **Multiple AI Models**: Support for OpenAI, Anthropic, DeepSeek, and more
- 📊 **Comprehensive Testing**: Full test coverage with pytest
- 🛠️ **Modern Python**: Type hints, dataclasses, and Python 3.9+ features
- 📦 **Easy Installation**: Modern packaging with pyproject.toml

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/username/ladder.git
cd ladder

# Install in development mode
make install-dev

# Or install directly
pip install -e ".[dev]"
```

### Basic Usage

#### Generate variants for a single integral

```python
import asyncio
from ladder import process_integral

async def main():
    integral = "integrate(x**2 + 2*x + 1, x)"
    difficulties = ["easier", "equivalent", "harder"]
    
    variants = await process_integral(
        integral, 
        difficulties, 
        num_variants=5
    )
    
    for variant in variants:
        print(f"Difficulty: {variant['requested_difficulty']}")
        print(f"Variant: {variant['variant']}")
        print(f"Verified: {variant['verification_passed']}")
        print("---")

asyncio.run(main())
```

#### Batch processing multiple integrals

```python
import asyncio
from ladder import BatchGenerator

async def main():
    integrals = [
        "integrate(sin(x), x)",
        "integrate(x**2, x)",
        "integrate(1/(x**2 + 1), x)"
    ]
    
    generator = BatchGenerator(
        batch_size=2,
        difficulties={"easier": 10, "equivalent": 5}
    )
    
    variants = await generator.process_all_integrals(
        integrals,
        output_dir="results",
        save_combined=True
    )
    
    generator.print_summary(variants)

asyncio.run(main())
```

## 🛠️ Development

### Setup Development Environment

```bash
# Install development dependencies
make install-dev

# Install pre-commit hooks
pre-commit install

# Run all checks
make check
```

### Code Quality

```bash
# Format code
make format

# Run linting  
make lint

# Run type checking
make type-check

# Run tests
make test

# Run tests with coverage
make test-cov
```

### Project Structure

```
ladder/
├── src/ladder/              # Main package
│   ├── __init__.py         # Package exports
│   ├── generate_variants.py # Core variant generation
│   ├── batch_generator.py  # Batch processing
│   ├── cli.py             # Command-line interface
│   ├── questions/         # Question datasets
│   └── utils/             # Utility modules
├── tests/                 # Test suite
├── docs/                  # Documentation  
├── pyproject.toml        # Modern Python packaging
├── Makefile              # Development commands
└── README.md            # This file
```

## 📚 API Reference

### Core Functions

#### `process_integral(integral_str, difficulties, num_variants=3)`

Generate variants for a single integral.

**Parameters:**
- `integral_str` (str): The integral expression in sympy format
- `difficulties` (List[str]): List of difficulty levels (`"easier"`, `"equivalent"`, `"harder"`)
- `num_variants` (int): Number of variants per difficulty level

**Returns:**
- `List[Dict]`: List of variant dictionaries with metadata

#### `BatchGenerator(batch_size=10, difficulties=None)`

Class for processing multiple integrals in batches.

**Parameters:**
- `batch_size` (int): Number of integrals to process concurrently
- `difficulties` (Dict[str, int]): Mapping of difficulty levels to variant counts

### Configuration

Set up API keys in environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export DEEPSEEK_API_KEY="your-deepseek-key"
# ... other API keys
```

## 🧪 Testing

The project includes comprehensive tests:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ladder --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Run the quality checks (`make check`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📖 Citation

If you use LADDER in your research, please cite:

```bibtex
@article{ladder2024,
  title={LADDER: Integration Problem Generator and Dataset Builder},
  author={Your Name},
  journal={arXiv preprint arXiv:2503.00735},
  year={2024}
}
```

## 🙏 Acknowledgments

- Built with modern Python packaging standards
- Inspired by mathematical problem generation research
- Uses various AI models for creative variant generation

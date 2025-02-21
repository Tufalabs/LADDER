# Integration Problem Generator and Dataset Builder

This project provides tools to generate and process mathematical integration problems, specifically designed for creating training datasets for machine learning models.

## Overview

The project consists of several components:
- Variant generation for integration problems
- Dataset formatting and processing
- Support for both JSON and Parquet output formats

## Components

### 1. Variant Generation

There are two main ways to generate integration problem variants:

#### Using batch_generate_variants.py
This script generates variations of integration problems using predefined patterns and rules.

```bash
python batch_generate_variants.py --output_dir variant_results --num_variants 100
```

Key parameters:
- `--output_dir`: Directory where variant JSON files will be saved
- `--num_variants`: Number of variants to generate per base problem

#### Using batch_tree_generator
This tool uses a tree-based approach to generate more complex integration problems.

```bash
python batch_tree_generator.py --output_dir tree_results --num_variants 100
```

Key parameters:

- `--output_dir`: Output directory for generated problems
- `--depth`: Maximum depth of the integration expression tree

### 2. Dataset Processing (format_data.py)

The `format_data.py` script processes the generated variants and creates structured datasets:

```bash
python utils/format_data.py
```




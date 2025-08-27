#!/usr/bin/env python3
"""
Modern batch variant generation script using the refactored LADDER package.

This script replaces the old batch_generate_variants.py with a cleaner,
more maintainable implementation using the new package structure.
"""

import argparse
import asyncio
from pathlib import Path

from src.ladder.batch_generator import BatchGenerator
from src.ladder.questions.mit_bee_regular_season_questions import BASE_QUESTIONS


async def main():
    """Main function for batch variant generation."""
    parser = argparse.ArgumentParser(
        description="Generate variants for multiple integration problems in batches"
    )
    parser.add_argument(
        "--output-dir",
        default="variant_results",
        help="Directory to save results (default: variant_results)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of integrals to process concurrently (default: 10)"
    )
    parser.add_argument(
        "--num-easier",
        type=int,
        default=30,
        help="Number of easier variants per integral (default: 30)"
    )
    parser.add_argument(
        "--num-equivalent",
        type=int,
        default=10,
        help="Number of equivalent variants per integral (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Create difficulties configuration
    difficulties = {
        "easier": args.num_easier,
        "equivalent": args.num_equivalent,
    }
    
    print(f"Processing {len(BASE_QUESTIONS)} questions")
    print(f"Batch size: {args.batch_size}")
    print(f"Difficulties: {difficulties}")
    print(f"Output directory: {args.output_dir}")
    
    # Create batch generator
    generator = BatchGenerator(
        batch_size=args.batch_size,
        difficulties=difficulties
    )
    
    # Process all integrals
    variants = await generator.process_all_integrals(
        BASE_QUESTIONS,
        output_dir=args.output_dir,
        save_individual=True,
        save_combined=True
    )
    
    # Print summary
    generator.print_summary(variants)


if __name__ == "__main__":
    asyncio.run(main())
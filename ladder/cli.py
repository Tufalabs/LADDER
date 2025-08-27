"""
Command-line interface for LADDER.

This module provides command-line utilities for generating integration problem variants
and processing batches of integrals.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import List, Optional

from ladder.batch_generator import BatchGenerator
from ladder.generate_variants import process_integral


def get_questions_from_module(module_name: str) -> List[str]:
    """
    Import questions from a module in the questions package.
    
    Args:
        module_name: Name of the module to import (e.g., 'mit_bee_regular_season_questions')
        
    Returns:
        List of question strings
        
    Raises:
        ImportError: If the module cannot be imported
        AttributeError: If the module doesn't have BASE_QUESTIONS
    """
    try:
        module_path = f"ladder.questions.{module_name}"
        module = __import__(module_path, fromlist=['BASE_QUESTIONS'])
        return getattr(module, 'BASE_QUESTIONS')
    except ImportError as e:
        raise ImportError(f"Could not import module {module_name}: {e}")
    except AttributeError as e:
        raise AttributeError(f"Module {module_name} does not have BASE_QUESTIONS attribute: {e}")


async def generate_variants() -> None:
    """CLI command for generating variants of a single integral."""
    parser = argparse.ArgumentParser(
        description="Generate variants for a single integration problem"
    )
    parser.add_argument(
        "integral",
        help="The integral expression to generate variants for"
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=["easier", "equivalent"],
        help="Difficulties to generate (default: easier equivalent)"
    )
    parser.add_argument(
        "--num-variants",
        type=int,
        default=3,
        help="Number of variants per difficulty (default: 3)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: variants.json)"
    )
    
    args = parser.parse_args()
    
    print(f"Generating variants for: {args.integral}")
    print(f"Difficulties: {args.difficulties}")
    print(f"Variants per difficulty: {args.num_variants}")
    
    try:
        variants = await process_integral(
            args.integral,
            args.difficulties,
            args.num_variants
        )
        
        output_file = args.output or "variants.json"
        with open(output_file, "w") as f:
            import json
            json.dump(variants, f, indent=2)
        
        print(f"\nGenerated {len(variants)} variants")
        print(f"Results saved to: {output_file}")
        
        # Print summary
        difficulty_counts = {}
        for variant in variants:
            diff = variant.get("requested_difficulty", "unknown")
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        
        print("\nSummary:")
        for diff, count in difficulty_counts.items():
            print(f"  {diff}: {count} variants")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


async def batch_generate() -> None:
    """CLI command for batch processing multiple integrals."""
    parser = argparse.ArgumentParser(
        description="Generate variants for multiple integration problems in batches"
    )
    parser.add_argument(
        "--questions-module",
        default="mit_bee_regular_season_questions",
        help="Module name containing BASE_QUESTIONS (default: mit_bee_regular_season_questions)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of integrals to process concurrently (default: 10)"
    )
    parser.add_argument(
        "--output-dir",
        default="variant_results",
        help="Directory to save results (default: variant_results)"
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=["easier", "equivalent"],
        help="Difficulties to generate (default: easier equivalent)"
    )
    parser.add_argument(
        "--num-variants",
        type=int,
        default=10,
        help="Number of variants per difficulty (default: 10)"
    )
    parser.add_argument(
        "--save-individual",
        action="store_true",
        help="Save individual batch results"
    )
    parser.add_argument(
        "--save-combined",
        action="store_true",
        default=True,
        help="Save combined results (default: True)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load questions
        print(f"Loading questions from: {args.questions_module}")
        questions = get_questions_from_module(args.questions_module)
        print(f"Loaded {len(questions)} questions")
        
        # Create difficulties dict
        difficulties = {diff: args.num_variants for diff in args.difficulties}
        
        # Create batch generator
        generator = BatchGenerator(
            batch_size=args.batch_size,
            difficulties=difficulties
        )
        
        print(f"Processing with batch size: {args.batch_size}")
        print(f"Difficulties: {difficulties}")
        print(f"Output directory: {args.output_dir}")
        
        # Process all integrals
        variants = await generator.process_all_integrals(
            questions,
            output_dir=args.output_dir,
            save_individual=args.save_individual,
            save_combined=args.save_combined
        )
        
        # Print summary
        generator.print_summary(variants)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main entry point for CLI."""
    if len(sys.argv) < 2:
        print("Usage: ladder-generate <integral> [options] OR ladder-batch [options]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "generate":
        sys.argv = ["ladder-generate"] + sys.argv[2:]  # Remove the "generate" part
        asyncio.run(generate_variants())
    elif command == "batch":
        sys.argv = ["ladder-batch"] + sys.argv[2:]  # Remove the "batch" part
        asyncio.run(batch_generate())
    else:
        # Assume it's an integral expression for single generation
        sys.argv = ["ladder-generate"] + sys.argv[1:]
        asyncio.run(generate_variants())


if __name__ == "__main__":
    main()
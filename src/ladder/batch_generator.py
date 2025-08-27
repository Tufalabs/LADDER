"""
Batch generator for processing multiple integration problems concurrently.

This module provides functionality to process multiple integrals in batches,
generating variants for each integral using the generate_variants functionality.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ladder.generate_variants import process_integral


class BatchGenerator:
    """
    Batch generator for processing multiple integration problems.
    
    Attributes:
        batch_size: Number of integrals to process concurrently
        difficulties: Dictionary mapping difficulty levels to number of variants
        max_retries: Maximum number of retries for failed batches
        retry_delay: Delay in seconds between retries
    """

    def __init__(
        self,
        batch_size: int = 10,
        difficulties: Optional[Dict[str, int]] = None,
        max_retries: int = 3,
        retry_delay: int = 5,
    ):
        """
        Initialize the BatchGenerator.
        
        Args:
            batch_size: Number of integrals to process concurrently
            difficulties: Dictionary mapping difficulty levels to number of variants
            max_retries: Maximum number of retries for failed batches
            retry_delay: Delay in seconds between retries
        """
        self.batch_size = batch_size
        self.difficulties = difficulties or {
            "easier": 30,
            "equivalent": 10,
        }
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    async def process_batch(self, integrals: List[str]) -> List[Dict[str, Any]]:
        """
        Process a batch of integrals concurrently with retry on failure.
        
        Args:
            integrals: List of integral strings to process
            
        Returns:
            List of processed variant dictionaries
            
        Raises:
            Exception: If processing fails after all retries
        """
        for attempt in range(self.max_retries):
            try:
                tasks = []
                for integral in integrals:
                    for difficulty in self.difficulties:
                        task = process_integral(
                            integral,
                            difficulties=[difficulty],
                            num_variants=self.difficulties[difficulty],
                        )
                        tasks.append(task)
                
                results = await asyncio.gather(*tasks)
                
                # Flatten the results
                flattened_results = []
                for result_group in results:
                    flattened_results.extend(result_group)
                
                return flattened_results
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    print(f"Retrying in {self.retry_delay} seconds...")
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise

    async def process_all_integrals(
        self,
        integrals: List[str],
        output_dir: str = "variant_results",
        save_individual: bool = True,
        save_combined: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Process all integrals in batches and save results.
        
        Args:
            integrals: List of all integrals to process
            output_dir: Directory to save results
            save_individual: Whether to save individual batch results
            save_combined: Whether to save combined results
            
        Returns:
            List of all processed variants
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        all_variants = []
        total_batches = (len(integrals) + self.batch_size - 1) // self.batch_size
        
        print(f"Processing {len(integrals)} integrals in {total_batches} batches")
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(integrals))
            batch_integrals = integrals[start_idx:end_idx]
            
            print(f"Processing batch {batch_idx + 1}/{total_batches} "
                  f"({len(batch_integrals)} integrals)")
            
            try:
                batch_variants = await self.process_batch(batch_integrals)
                all_variants.extend(batch_variants)
                
                if save_individual:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    batch_filename = f"variants_batch_{batch_idx + 1}_{timestamp}.json"
                    batch_filepath = output_path / batch_filename
                    
                    with open(batch_filepath, "w") as f:
                        json.dump(batch_variants, f, indent=2)
                    
                    print(f"Saved batch {batch_idx + 1} results to {batch_filepath}")
                
            except Exception as e:
                print(f"Failed to process batch {batch_idx + 1}: {e}")
                continue
        
        if save_combined:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            combined_filename = f"variants_all_{timestamp}.json"
            combined_filepath = output_path / combined_filename
            
            with open(combined_filepath, "w") as f:
                json.dump(all_variants, f, indent=2)
            
            print(f"Saved all results to {combined_filepath}")
        
        print(f"Processing complete. Generated {len(all_variants)} total variants.")
        return all_variants

    def print_summary(self, variants: List[Dict[str, Any]]) -> None:
        """
        Print a summary of the generated variants.
        
        Args:
            variants: List of variant dictionaries
        """
        if not variants:
            print("No variants generated.")
            return
        
        # Count by difficulty
        difficulty_counts = {}
        verification_stats = {"passed": 0, "failed": 0, "unknown": 0}
        
        for variant in variants:
            difficulty = variant.get("requested_difficulty", "unknown")
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
            
            verification = variant.get("verification_passed")
            if verification is True:
                verification_stats["passed"] += 1
            elif verification is False:
                verification_stats["failed"] += 1
            else:
                verification_stats["unknown"] += 1
        
        print("\n=== SUMMARY ===")
        print(f"Total variants: {len(variants)}")
        print("\nBy difficulty:")
        for difficulty, count in difficulty_counts.items():
            print(f"  {difficulty}: {count}")
        
        print("\nVerification status:")
        for status, count in verification_stats.items():
            percentage = (count / len(variants)) * 100
            print(f"  {status}: {count} ({percentage:.1f}%)")


async def main() -> None:
    """Example usage of BatchGenerator."""
    # Example integrals for testing
    test_integrals = [
        "integrate(1/(x**2 - x + 1), (x, 0, 1))",
        "integrate(x**2 + 2*x + 1, x)",
        "integrate(sin(x)*cos(x), x)",
    ]
    
    generator = BatchGenerator(batch_size=2)
    variants = await generator.process_all_integrals(
        test_integrals,
        output_dir="test_results",
    )
    
    generator.print_summary(variants)


if __name__ == "__main__":
    asyncio.run(main())
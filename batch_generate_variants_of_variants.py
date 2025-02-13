"""
batch_generate_variants.py

This script processes multiple integrals from incorrect_questions.py in batches,
generating easier and equivalent variants for each integral using the generate_variants.py
functionality.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from generate_variants import process_integral
import random

def load_questions_from_json(max_questions=1000):
    """
    Load questions from integration_train.json file and return a random subset.
    Args:
        max_questions: Maximum number of questions to return
    """
    json_path = Path("output/integration_train.json")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract the integration problems from the prompts
    questions = []
    for item in data:
        if isinstance(item, dict) and 'prompt' in item:
            for prompt in item['prompt']:
                if prompt.get('role') == 'user':
                    # Extract the integral from the content
                    content = prompt.get('content', '')
                    if 'integrate(' in content:
                        integral = content.split('integrate(')[1].split(')')[0]
                        questions.append(integral)
    
    # Select a random subset if we have more questions than max_questions
    if len(questions) > max_questions:
        questions = random.sample(questions, max_questions)
    
    print(f"Loaded {len(questions)} questions for processing")
    return questions

# Replace QUESTIONS definition
QUESTIONS = load_questions_from_json()

BATCH_SIZE = 10  # Number of integrals to process concurrently
DIFFICULTIES = {
    "easier": 20,  # Number of easier variants to generate
    "equivalent": 20,  # Number of equivalent variants to generate
    "harder": 10  # Number of harder variants to generate
}

async def process_batch(integrals: List[str]) -> List[Dict]:
    """Process a batch of integrals concurrently with retry on failure."""
    max_retries = 3  # Maximum number of retries
    retry_delay = 5  # Delay in seconds before retrying

    for attempt in range(max_retries):
        try:
            tasks = [
                process_integral(
                    integral,
                    difficulties=list(DIFFICULTIES.keys()),  # Pass the keys of the difficulties
                    num_variants=DIFFICULTIES[difficulty]  # Pass the number of variants for each difficulty
                )
                for integral in integrals
                for difficulty in DIFFICULTIES  # Iterate over each difficulty
            ]
            results = await asyncio.gather(*tasks)
            return results
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)  # Wait before retrying
            else:
                raise  # Re-raise the last exception if max retries reached

async def main():
    # Create output directory if it doesn't exist
    output_dir = Path("variant_results")
    output_dir.mkdir(exist_ok=True)
    
    # Process integrals in batches
    all_results = []
    total_integrals = len(QUESTIONS)
    shuffled_questions = QUESTIONS.copy()  # Create a copy of the list
    random.shuffle(shuffled_questions)  # Shuffle the copy
    for i in range(0, total_integrals, BATCH_SIZE):
        batch = shuffled_questions[i:i + BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE + 1}/{(total_integrals + BATCH_SIZE - 1)//BATCH_SIZE}")
        print(f"Integrals {i+1}-{min(i+BATCH_SIZE, total_integrals)} of {total_integrals}")
        
        batch_results = await process_batch(batch)
        all_results.extend(batch_results)
        
        # Save intermediate results after each batch
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        batch_filename = output_dir / f"variants_batch_{i//BATCH_SIZE + 1}_{timestamp}.json"
        with open(batch_filename, "w") as f:
            json.dump(batch_results, f, indent=2)
        
        print(f"Batch results saved to {batch_filename}")
    
    # Save final combined results
    final_filename = output_dir / f"variants_all_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(final_filename, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll results saved to {final_filename}")
    
    # Print summary statistics
    total_variants = sum(len(result) for result in all_results)
    print(f"\nSummary:")
    print(f"Total integrals processed: {total_integrals}")
    print(f"Total variants generated: {total_variants}")
    print(f"Average variants per integral: {total_variants/total_integrals:.2f}")

if __name__ == "__main__":
    asyncio.run(main())     
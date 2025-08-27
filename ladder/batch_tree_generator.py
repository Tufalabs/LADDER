#!/usr/bin/env python3
"""
batch_tree_generator.py

This script builds separate question trees for a subset of the base questions.
For each selected base question, it:
  - Builds an initial tree using process_question (from question_tree_manager.py)
  - Recursively generates additional variants (with a recursion depth of 3)
    so that the final tree has roughly around 500 nodes.
  - Saves each tree to a separate JSON file in the specified output directory.

Usage:
    $ python batch_tree_generator.py --num-base 5 --num-variants 10 --depth 3 --output-dir trees
"""

import asyncio
import argparse
from pathlib import Path
import time
import random

# Import required functions and the base questions list.
from question_tree_manager import process_question, generate_recursive_variants, save_tree, print_tree_analytics
from incorrect_hardest_questions import QUESTIONS_INCORRECT as QUESTIONS

async def generate_tree_for_question(question: str, num_variants: int, recursion_depth: int):
    """
    Generate a question tree for a single base question.
    
    The tree is built by first processing the question to generate a base tree,
    then recursively generating additional easier variants with the specified depth.
    """
    print(f"[Tree Generator] Processing base question: {question}")
    # Build the initial tree (a single QuestionGroup) for the base question.
    tree = [await process_question(question)]
    
    # Generate additional variants recursively.
    # With recursion_depth=3 and num_variants=10, the resulting tree should have roughly ~500 nodes.
    await generate_recursive_variants(
        tree,
        starting_level=0,       # Starting from the root group.
        recursion_depth=recursion_depth,
        num_variants=num_variants
    )
    return tree

async def main():
    parser = argparse.ArgumentParser(
        description="Build separate question trees for a subset of base questions. "
                    "Each tree will have a recursion depth of 3 and roughly ~500 nodes."
    )
    parser.add_argument("-n", "--num-base", type=int, default=20,
                        help="Number of base questions to process (from the start of the list)")
    parser.add_argument("-v", "--num-variants", type=int, default=6,
                        help="Number of new variants to generate per parent (passed to process_integral)")
    parser.add_argument("-d", "--depth", type=int, default=3,
                        help="Recursion depth for generating variants")
    parser.add_argument("-o", "--output-dir", type=str, default="trees",
                        help="Output directory to save the trees")
 
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select the first N base questions.
    selected_questions = QUESTIONS[:args.num_base]
    random.shuffle(selected_questions)  # Randomize the order of questions
    print(f"[Main] Selected {len(selected_questions)} base question(s) for processing.\n")

    # Process each selected question.
    for i, question in enumerate(selected_questions, start=1):
        print(f"\n=== Generating tree for base question {i} ===")
        tree = await generate_tree_for_question(question, num_variants=args.num_variants, recursion_depth=args.depth)
        timestamp = int(time.time() * 1000)  # Get current timestamp in milliseconds
        output_file = output_dir / f"tree_{timestamp}.json"
        save_tree(tree, output_file, base_question=question)
        print(f"[Main] Tree for base question {i} saved to {output_file}\n")
        print("[Main] Tree analytics:")
        print_tree_analytics(tree)
        # Uncomment the following line if you wish to print the entire tree structure.
        # from question_tree_manager import print_tree
        # print_tree(tree)

if __name__ == "__main__":
    asyncio.run(main())

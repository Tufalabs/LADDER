#!/usr/bin/env python3
"""
question_tree_manager.py

This script manages a tree-structure of questions.
Each node (a "question group") contains a list of variants
(all at the same numeric level) and, if applicable, a list of child groups,
which represent easier variants generated from the parent.

It provides functionality to:
  - Build an initial tree from the base questions (level 0),
    generating "equivalent" and "easier" variants concurrently using batching.
  - Save and load the tree to/from JSON so that the level is clearly stored.
  - Generate additional variants at a specified level:
      • For level 0, additional "equivalent" variants are added to the root group.
      • For levels > 0, the script finds parent groups (at level L–1)
        and then generates new easier variants as child nodes.
  - Print tree analytics (node counts per level).
  - Recursively generate additional easier variants (with configurable depth).
  - Prune the tree to remove duplicate nodes (nodes with the same primary variant).

Optional Web Interface:
  You might serve the JSON output with a lightweight Flask or Streamlit app,
  and use a JavaScript library (e.g. D3.js or dash_cytoscape) to render the tree.
  (This code only manages the data; the web front-end is up to you.)

Usage examples (via the command line):

  • Build the initial tree and save it:
      $ python question_tree_manager.py build --output questions_tree.json [--incremental]

  • Generate more variants at level 1 (from level 0 parents):
      $ python question_tree_manager.py generate 1 --input questions_tree.json --output questions_tree.json --num 5 --max-parents 10 [--incremental]

  • Recursively generate variants (e.g., generate 3 deeper levels starting from level 0):
      $ python question_tree_manager.py generate_recursive 0 3 --input questions_tree.json --output questions_tree.json --num 5 --max-parents 10 [--incremental]

  • Prune the tree to remove duplicate nodes:
      $ python question_tree_manager.py prune --input questions_tree.json --output questions_tree.json

  • Print tree analytics:
      $ python question_tree_manager.py analytics --input questions_tree.json

  • Print the current tree:
      $ python question_tree_manager.py print --input questions_tree.json
"""

import asyncio
import json
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import random  # Useful for sampling if needed

# ------------------------- Configuration Constants ---------------------------
DEFAULT_EQUIV_VARIANTS = 10         # If set to 0, skip equivalent generation and use easier variant alone.
DEFAULT_EASIER_VARIANTS = 10        # Number of easier variants to generate per question.
BATCH_SIZE = 30                    # Maximum number of concurrent tasks in a batch.
DEFAULT_MAX_PARENT_SAMPLE = 30     # Maximum number of parent nodes to sample from on a level (<=0 means all).
DEFAULT_RECURSION_DEPTH = 6        # Default recursion depth for recursive generation.
DEFAULT_START_LEVEL = 0            # Default starting level for recursive generation.
# -----------------------------------------------------------------------------

# Import your base questions and variant generator.
from questions.mit_bee_qualifiying_questions import BASE_QUESTIONS as QUESTIONS
from generate_variants import process_integral

# -----------------------------------------------------------------------------
# Helper function to run tasks in batches concurrently.
# An optional callback (incremental_callback) is invoked after each batch.
# -----------------------------------------------------------------------------

async def run_tasks_in_batches(tasks: List[asyncio.coroutine],
                               batch_size: int = BATCH_SIZE,
                               incremental_callback=None):
    total = len(tasks)
    results = []
    batch_num = 1
    for i in range(0, total, batch_size):
        print(f"[Batch Runner] Processing batch {batch_num} (tasks {i+1} to {min(i+batch_size, total)}) out of {total} tasks...")
        batch = tasks[i:i+batch_size]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)
        if incremental_callback is not None:
            await incremental_callback(results, batch_num)
        print(f"[Batch Runner] Completed batch {batch_num}")
        batch_num += 1
    return results

# -----------------------------------------------------------------------------
# Data structure for the question tree.
# -----------------------------------------------------------------------------

@dataclass
class QuestionGroup:
    level: int
    variants: List[str]
    children: List['QuestionGroup'] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "variants": self.variants,
            "children": [child.to_dict() for child in self.children],
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'QuestionGroup':
        children = [cls.from_dict(child) for child in d.get("children", [])]
        return cls(level=d["level"], variants=d["variants"], children=children)

# -----------------------------------------------------------------------------
# Helper functions for traversing and saving the tree.
# -----------------------------------------------------------------------------
def get_variant_text(item):
    """
    If item is a dict (returned by process_integral), extract the actual variant string.
    Otherwise, assume it is already a string.
    """
    if isinstance(item, dict):
        return item.get("variant", item.get("original", str(item)))
    return item

def find_groups_at_level(groups: List[QuestionGroup], level: int) -> List[QuestionGroup]:
    """Recursively search the tree for groups at the given level."""
    result = []
    for group in groups:
        if group.level == level:
            result.append(group)
        result.extend(find_groups_at_level(group.children, level))
    return result

def save_tree(tree: List[QuestionGroup], filename: Path, base_question: str = None):
    """Save the tree as JSON with an optional base question."""
    data = {
        "base_question": base_question,
        "tree": [group.to_dict() for group in tree]
    } if base_question else [group.to_dict() for group in tree]
    
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[SAVE] Tree saved to {filename}")

def load_tree(filename: Path) -> List[QuestionGroup]:
    """Load the tree from JSON."""
    with open(filename, "r") as f:
        data = json.load(f)
    # Handle both new format (with base_question) and old format
    tree_data = data["tree"] if isinstance(data, dict) else data
    tree = [QuestionGroup.from_dict(item) for item in tree_data]
    print(f"[LOAD] Loaded tree from {filename}")
    return tree

def print_tree(tree: List[QuestionGroup], indent: int = 0):
    """Print the tree structure for visual inspection."""
    pad = "  " * indent
    for group in tree:
        print(f"{pad}Level {group.level}:")
        for i, q in enumerate(group.variants, 1):
            print(f"{pad}  [{i}] {q}")
        if group.children:
            print(f"{pad}  Easier variants:")
            print_tree(group.children, indent=indent+2)

# -----------------------------------------------------------------------------
# Analytics: Collect and print node counts per level.
# -----------------------------------------------------------------------------

def collect_tree_analytics(groups: List[QuestionGroup], level: int = 0, analytics: Dict[int, int] = None) -> Dict[int, int]:
    if analytics is None:
        analytics = {}
    for group in groups:
        analytics[level] = analytics.get(level, 0) + 1
        if group.children:
            collect_tree_analytics(group.children, level+1, analytics)
    return analytics

def print_tree_analytics(tree: List[QuestionGroup]):
    analytics = collect_tree_analytics(tree)
    print("[Analytics] Tree node counts per level:")
    for lvl in sorted(analytics.keys()):
        print(f"  Level {lvl}: {analytics[lvl]} node(s)")
    total = sum(analytics.values())
    print(f"  Total nodes: {total}")

# -----------------------------------------------------------------------------
# Functions to build the tree and generate additional variants.
# -----------------------------------------------------------------------------

async def process_question(q: str) -> QuestionGroup:
    print(f"[Process Question] Processing base question: {q}")
    try:
        equiv = await process_integral(q, difficulties=["equivalent"], num_variants=DEFAULT_EQUIV_VARIANTS)
        print(f"[Process Question] For question '{q}', equivalent output: {equiv}")
    except Exception as e:
        print(f"[Process Question] Error generating equivalent variants for {q}: {e}")
        equiv = []
    # Extract text from each equivalent variant.
    root_variants = [q] + [get_variant_text(v) for v in equiv]

    try:
        easier = await process_integral(q, difficulties=["easier"], num_variants=DEFAULT_EASIER_VARIANTS)
        print(f"[Process Question] For question '{q}', easier output: {easier}")
    except Exception as e:
        print(f"[Process Question] Error generating easier variants for {q}: {e}")
        easier = []

    async def process_easier_variant(ev) -> QuestionGroup:
        # Ensure we work with a text version.
        ev_text = get_variant_text(ev)
        print(f"  [Process Easier] Processing easier variant: {ev_text}")
        if DEFAULT_EQUIV_VARIANTS > 0:
            try:
                equiv_child = await process_integral(ev_text, difficulties=["equivalent"], num_variants=DEFAULT_EQUIV_VARIANTS)
                print(f"  [Process Easier] For easier variant '{ev_text}', equivalent output: {equiv_child}")
            except Exception as e:
                print(f"  [Process Easier] Error generating equivalent variants for easier variant {ev_text}: {e}")
                equiv_child = []
            child_variants = [ev_text] + [get_variant_text(v) for v in equiv_child]
        else:
            child_variants = [ev_text]
            print("  [Process Easier] DEFAULT_EQUIV_VARIANTS is 0; using easier variant as node without additional equivalents.")
        return QuestionGroup(level=1, variants=child_variants, children=[])
    
    child_groups = []
    if easier:
        print(f"[Process Question] Processing {len(easier)} easier variants concurrently for question: {q}")
        tasks = [process_easier_variant(ev) for ev in easier]
        child_groups = await run_tasks_in_batches(tasks, BATCH_SIZE)
    
    print(f"[Process Question] Finished processing question: {q} (Level 0 variants: {len(root_variants)}, Child groups: {len(child_groups)})")
    return QuestionGroup(level=0, variants=root_variants, children=child_groups)

async def add_equivalent_variants(group: QuestionGroup, num_variants: int):
    print(f"[Add Equivalent] Generating additional equivalent variants for: {group.variants[0]} (level {group.level})")
    try:
        new_equiv = await process_integral(group.variants[0], difficulties=["equivalent"], num_variants=num_variants)
        print(f"[Add Equivalent] Generated {len(new_equiv)} new equivalent variants for: {group.variants[0]}")
    except Exception as e:
        print(f"[Add Equivalent] Error generating equivalent variants for {group.variants[0]}: {e}")
        new_equiv = []
    if new_equiv:
        group.variants.extend(new_equiv)
        print(f"[Add Equivalent] Added {len(new_equiv)} new equivalent variants to the group.")
# In process_easier_for_parent: also extract text before using as input.
async def process_easier_for_parent(parent: QuestionGroup, target_level: int, num_variants: int):
    parent_input = get_variant_text(parent.variants[0])
    print(f"[Process Easier Parent] Generating easier variants for parent question: {parent_input} (level {parent.level})")
    try:
        new_easier = await process_integral(parent_input, difficulties=["easier"], num_variants=num_variants)
        print(f"[Process Easier Parent] For parent '{parent_input}', easier output: {new_easier}")
    except Exception as e:
        print(f"[Process Easier Parent] Error generating easier variants for {parent_input}: {e}")
        new_easier = []
    if not new_easier:
        fallback = parent_input
        new_easier = [fallback]
        print(f"[Process Easier Parent] No easier variants generated; using fallback: {fallback}")
    
    async def process_new_easier(ev) -> QuestionGroup:
        ev_text = get_variant_text(ev)
        print(f"  [Process New Easier] Processing new easier variant: {ev_text}")
        if DEFAULT_EQUIV_VARIANTS > 0:
            try:
                new_equiv = await process_integral(ev_text, difficulties=["equivalent"], num_variants=DEFAULT_EQUIV_VARIANTS)
                print(f"  [Process New Easier] For new easier variant '{ev_text}', equivalent output: {new_equiv}")
            except Exception as e:
                print(f"  [Process New Easier] Error generating equivalent variants for easier variant {ev_text}: {e}")
                new_equiv = []
            child_variants = [ev_text] + [get_variant_text(v) for v in new_equiv]
        else:
            child_variants = [ev_text]
            print("  [Process New Easier] DEFAULT_EQUIV_VARIANTS is 0; using easier variant as node without additional equivalents.")
        return QuestionGroup(level=target_level, variants=child_variants, children=[])
    
    tasks = [process_new_easier(ev) for ev in new_easier]
    new_child_groups = await run_tasks_in_batches(tasks, BATCH_SIZE)
    parent.children.extend(new_child_groups)
    print(f"[Process Easier Parent] Added {len(new_child_groups)} new child group(s) to parent question: {parent_input}")

async def generate_more_variants(tree: List[QuestionGroup],
                                 target_level: int,
                                 num_variants: int,
                                 max_parents: int = DEFAULT_MAX_PARENT_SAMPLE,
                                 incremental: bool = False,
                                 incremental_prefix: str = None):
    print(f"[Generate More] Starting generation of additional variants at target level {target_level} …")
    if target_level == 0:
        tasks = [add_equivalent_variants(group, num_variants) for group in tree]
        if incremental and incremental_prefix:
            async def inc_callback(results, batch_num):
                filename = Path(f"{incremental_prefix}_partial_batch_{batch_num}.json")
                print(f"[Incremental Save] Saving updated tree after batch {batch_num} to {filename}")
                save_tree(tree, filename)
            await run_tasks_in_batches(tasks, BATCH_SIZE, incremental_callback=inc_callback)
        else:
            await run_tasks_in_batches(tasks, BATCH_SIZE)
    else:
        parent_groups = find_groups_at_level(tree, target_level - 1)
        if max_parents > 0:
            parent_groups = parent_groups[:max_parents]
        if not parent_groups:
            print(f"[Generate More] No parent groups found at level {target_level-1}; nothing to do.")
            return
        tasks = [process_easier_for_parent(parent, target_level, num_variants) for parent in parent_groups]
        if incremental and incremental_prefix:
            async def inc_callback(results, batch_num):
                filename = Path(f"{incremental_prefix}_partial_batch_{batch_num}.json")
                print(f"[Incremental Save] Saving updated tree after batch {batch_num} to {filename}")
                save_tree(tree, filename)
            await run_tasks_in_batches(tasks, BATCH_SIZE, incremental_callback=inc_callback)
        else:
            await run_tasks_in_batches(tasks, BATCH_SIZE)
    print(f"[Generate More] Finished generating additional variants at level {target_level}.")

async def generate_recursive_variants(tree: List[QuestionGroup],
                                      starting_level: int,
                                      recursion_depth: int,
                                      num_variants: int,
                                      max_parents: int = DEFAULT_MAX_PARENT_SAMPLE,
                                      incremental: bool = False,
                                      incremental_prefix: str = None):
    """
    Recursively generate additional easier variants.
    For example, if starting_level=0 and recursion_depth=3, then new variants
    will be generated for level 1, then level 2, then level 3.
    If the requested starting level is not found in the tree,
    the function warns and uses the maximum available level.
    """
    analytics = collect_tree_analytics(tree)
    if starting_level not in analytics:
        available = max(analytics.keys())
        print(f"[Recursive Generate] Warning: starting level {starting_level} not found. Using maximum available level {available} as starting level.")
        current_level = available
    else:
        current_level = starting_level
    print(f"[Recursive Generate] Starting recursive generation from level {current_level} with recursion depth {recursion_depth}.")
    for d in range(recursion_depth):
        target_level = current_level + 1
        print(f"[Recursive Generate] Generating variants for level {target_level} (from parent level {current_level})...")
        await generate_more_variants(tree, target_level, num_variants, max_parents, incremental, incremental_prefix)
        new_nodes = find_groups_at_level(tree, target_level)
        print(f"[Recursive Generate] Level {target_level} now has {len(new_nodes)} node(s).")
        current_level = target_level
    print("[Recursive Generate] Finished recursive generation.")

# -----------------------------------------------------------------------------
# Prune function: Remove duplicate nodes based on the primary variant.
# -----------------------------------------------------------------------------

def unique_list(lst: List[str]) -> List[str]:
    seen = set()
    unique = []
    for item in lst:
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return unique

def prune_tree_nodes(nodes: List[QuestionGroup], seen: set = None) -> List[QuestionGroup]:
    if seen is None:
        seen = set()
    pruned = []
    for node in nodes:
        # Remove duplicate variants within the node.
        node.variants = unique_list(node.variants)
        primary = node.variants[0] if node.variants else None
        if primary is not None:
            if primary in seen:
                print(f"[Prune] Removing duplicate node with primary variant: {primary}")
                continue
            seen.add(primary)
        # Recursively prune children.
        node.children = prune_tree_nodes(node.children, seen)
        pruned.append(node)
    return pruned

def prune_tree(tree: List[QuestionGroup]) -> List[QuestionGroup]:
    """Return a new tree with duplicate nodes (by primary variant) removed."""
    return prune_tree_nodes(tree)

async def build_initial_tree(incremental: bool = False, incremental_prefix: str = None) -> List[QuestionGroup]:
    """Build the initial tree from base QUESTIONS using batched concurrent processing."""
    print("[Build Tree] Building initial tree …")
    tasks = [process_question(q) for q in QUESTIONS]
    if incremental and incremental_prefix:
        async def inc_callback(results, batch_num):
            filename = Path(f"{incremental_prefix}_partial_batch_{batch_num}.json")
            print(f"[Incremental Save] Saving partial tree after batch {batch_num} to {filename}")
            save_tree(results, filename)
        tree = await run_tasks_in_batches(tasks, BATCH_SIZE, incremental_callback=inc_callback)
    else:
        tree = await run_tasks_in_batches(tasks, BATCH_SIZE)
    print("[Build Tree] Finished building initial tree.")
    return tree

# -----------------------------------------------------------------------------
# Command-line interface.
# -----------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Manage a tree of question variants with numeric levels using batched processing, tree analytics, recursive generation, and pruning."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build initial tree from base questions")
    build_parser.add_argument("--output", type=str, default="questions_tree.json", help="Output JSON file")
    build_parser.add_argument("--incremental", action="store_true", help="Enable incremental saving along the way")

    generate_parser = subparsers.add_parser("generate", help="Generate more variants at a target level")
    generate_parser.add_argument("level", type=int, help="Target level for new variants (0 for equivalent, >0 for easier)")
    generate_parser.add_argument("--input", type=str, default="questions_tree.json", help="Input JSON file")
    generate_parser.add_argument("--output", type=str, default="questions_tree.json", help="Output JSON file (will be overwritten)")
    generate_parser.add_argument("--num", type=int, default=DEFAULT_EQUIV_VARIANTS, help="Number of new variants to generate per parent")
    generate_parser.add_argument("--max-parents", type=int, default=DEFAULT_MAX_PARENT_SAMPLE, help="Maximum number of parent nodes to sample from on a level (<=0 means all)")
    generate_parser.add_argument("--incremental", action="store_true", help="Enable incremental saving along the way")

    rec_parser = subparsers.add_parser("generate_recursive", help="Recursively generate additional easier variants.")
    rec_parser.add_argument("start_level", type=int, default=DEFAULT_START_LEVEL, nargs="?", help="Starting level (e.g., 0 if generating from root)")
    rec_parser.add_argument("depth", type=int, default=DEFAULT_RECURSION_DEPTH, nargs="?", help="How many additional levels to generate recursively")
    rec_parser.add_argument("--input", type=str, default="questions_tree.json", help="Input JSON file")
    rec_parser.add_argument("--output", type=str, default="questions_tree.json", help="Output JSON file (will be overwritten)")
    rec_parser.add_argument("--num", type=int, default=DEFAULT_EQUIV_VARIANTS, help="Number of new variants to generate per parent")
    rec_parser.add_argument("--max-parents", type=int, default=DEFAULT_MAX_PARENT_SAMPLE, help="Maximum number of parent nodes to sample from on a level (<=0 means all)")
    rec_parser.add_argument("--incremental", action="store_true", help="Enable incremental saving along the way")

    prune_parser = subparsers.add_parser("prune", help="Prune the tree to remove duplicate nodes by primary variant")
    prune_parser.add_argument("--input", type=str, default="questions_tree.json", help="Input JSON file")
    prune_parser.add_argument("--output", type=str, default="questions_tree_pruned.json", help="Output JSON file for the pruned tree")

    analytics_parser = subparsers.add_parser("analytics", help="Print tree analytics (node counts per level)")
    analytics_parser.add_argument("--input", type=str, default="questions_tree.json", help="Input JSON file")

    print_parser = subparsers.add_parser("print", help="Print the current tree")
    print_parser.add_argument("--input", type=str, default="questions_tree.json", help="Input JSON file")

    args = parser.parse_args()

    if args.command == "build":
        tree = await build_initial_tree(incremental=args.incremental, incremental_prefix=args.output)
        out_file = Path(args.output)
        save_tree(tree, out_file)
        print("[Main] Initial tree built. Here is a summary:")
        print_tree(tree)

    elif args.command == "generate":
        tree = load_tree(Path(args.input))
        await generate_more_variants(tree, target_level=args.level, num_variants=args.num,
                                     max_parents=args.max_parents,
                                     incremental=args.incremental, incremental_prefix=args.output)
        save_tree(tree, Path(args.output))
        print("[Main] Updated tree:")
        print_tree(tree)

    elif args.command == "generate_recursive":
        tree = load_tree(Path(args.input))
        await generate_recursive_variants(tree,
                                          starting_level=args.start_level,
                                          recursion_depth=args.depth,
                                          num_variants=args.num,
                                          max_parents=args.max_parents,
                                          incremental=args.incremental,
                                          incremental_prefix=args.output)
        save_tree(tree, Path(args.output))
        print("[Main] Updated tree after recursive generation:")
        # Uncomment the following line to print the entire tree:
        # print_tree(tree)

    elif args.command == "prune":
        tree = load_tree(Path(args.input))
        pruned = prune_tree(tree)
        save_tree(pruned, Path(args.output))
        print("[Main] Pruned tree:")
        # print_tree(pruned)

    elif args.command == "analytics":
        tree = load_tree(Path(args.input))
        print_tree_analytics(tree)

    elif args.command == "print":
        tree = load_tree(Path(args.input))
        # Uncomment the following line to print the entire tree:
        # print_tree(tree)

if __name__ == "__main__":
    asyncio.run(main())

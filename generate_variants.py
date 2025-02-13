"""
self_improve.py

An experimental pipeline that:
- Accepts an integral (or uses a preset one) and a set of difficulty targets.
- For each requested difficulty (e.g., easier, equivalent, or harder) it generates several variants.
- Each LLM prompt now generates up to 10 variants at once. If the user requests more than 10 variants,
  the work is split into multiple concurrent calls.
- Each variant's antiderivative is attempted symbolically via Sympy.
- A second LLM prompt asks for a difficulty evaluation (easier/harder/equivalent) as a double-check.
- The antiderivative solution (i.e. the integration result) is computed, and it is evaluated at three random points.
- Points that produce complex values are skipped.
- Variants judged as "harder" are filtered out when not desired.
- All results are saved to "variants.json".

If the integration (antiderivative computation) takes more than 5 seconds, it is skipped and the variant is
returned with a solution of None (and an empty evaluation dictionary), rather than being thrown out.
"""

import asyncio
import json
import math
import random
import re
from datetime import datetime
import concurrent.futures
import sympy as sp

MODEL = "gpt-4o-mini"
TIMEOUT_SECONDS = 1  # Maximum allowed seconds for integration

# New flag: if set to False, we will not compute the symbolic solution
CALCULATE_SYMBOLIC = False  # Set to False to disable symbolic integration computation.

# Import our LLM-based generation function from the provided utils.inference module.
from utils.inference import generate_text

# Create a global process pool executor to reuse worker processes.
executor = concurrent.futures.ProcessPoolExecutor(max_workers=4)

def integrate_wrapper(integrand, x):
    """
    A top-level function to compute sp.integrate(integrand, x).
    This function is picklable and can be used with the executor.
    """
    return sp.integrate(integrand, x)

def run_integration(integrand, x, timeout=TIMEOUT_SECONDS):
    """
    Run sp.integrate(integrand, x) using a persistent process pool.
    If it takes longer than 'timeout' seconds, a TimeoutError is raised.
    """
    future = executor.submit(integrate_wrapper, integrand, x)
    return future.result(timeout=timeout)

def verify_integral(integral_str: str) -> bool:
    """
    Verify the integral by checking that the derivative of the antiderivative equals the original integrand.
    Uses symbolic simplification to check for an exact zero difference.
    Note: In this revised version we no longer use this result to drop variants.
    """
    x = sp.symbols('x')
    try:
        pattern = r"integrate\((.+),\s*x\)"
        match = re.search(pattern, integral_str)
        if not match:
            return False

        integrand_str = match.group(1)
        integrand = sp.sympify(integrand_str)
        try:
            antideriv = run_integration(integrand, x, timeout=TIMEOUT_SECONDS)
        except Exception as e:
            print("Integration timed out in verify_integral; returning non-verified result.")
            return False

        diff_expr = sp.simplify(sp.diff(antideriv, x) - integrand)
        return diff_expr == 0
    except Exception as e:
        print("Error verifying integral:", e)
        return False

def compute_solution_and_evals(integral_str: str, num_points: int = 3, lower: float = -10, upper: float = 10, tol: float = 1e-6):
    """
    Given an integral string of the form "integrate(<integrand>, x)", compute the antiderivative (solution)
    and evaluate that solution at up to `num_points` random values of x.
    If the antiderivative cannot be computed within TIMEOUT_SECONDS, returns None and an empty dict.
    
    Returns:
        solution_str: The antiderivative as a string (or None if not computable or if symbolic integration is disabled).
        evaluations: A dictionary mapping each random x value (rounded) to the numerical evaluation.
    """
    # If symbolic integration is disabled, immediately return None
    if not CALCULATE_SYMBOLIC:
        return None, {}

    x = sp.symbols('x')
    try:
        pattern = r"integrate\((.+),\s*x\)"
        match = re.search(pattern, integral_str)
        if not match:
            return None, {}

        integrand_str = match.group(1)
        integrand = sp.sympify(integrand_str)
        try:
            antideriv = run_integration(integrand, x, timeout=TIMEOUT_SECONDS)
        except Exception as e:
            print("Integration timed out in compute_solution_and_evals; marking as too hard.")
            return None, {}
        solution_str = str(antideriv)
        evaluations = {}
        attempts = 0
        max_attempts = num_points * 10
        while len(evaluations) < num_points and attempts < max_attempts:
            attempts += 1
            test_val = random.uniform(lower, upper)
            eval_val = antideriv.evalf(subs={x: test_val})
            if hasattr(eval_val, "as_real_imag"):
                re_val, im_val = eval_val.as_real_imag()
                if abs(im_val) < tol:
                    evaluations[round(test_val, 3)] = float(re_val)
            else:
                evaluations[round(test_val, 3)] = float(eval_val)
        return solution_str, evaluations
    except Exception as e:
        print("Error computing solution/evaluations:", e)
        return None, {}

def parse_variants(text: str) -> list:
    """
    Parse the LLM response text and extract a list of variant dictionaries.
    The expected format for each variant is:
    
    ====
    Variant <number>:
    Reasoning: <explanation>
    Variant: integrate(<integrand>, x)
    ====
    """
    variants = []
    blocks = re.split(r"====\s*", text)
    for block in blocks:
        if "Variant:" in block and "Reasoning:" in block:
            reasoning_match = re.search(r"Reasoning:\s*(.*?)\s*Variant:", block, re.DOTALL)
            variant_match = re.search(r"Variant:\s*(integrate\([^,]+,\s*x\))", block)
            if variant_match:
                variant_expr = variant_match.group(1).strip()
                reasoning_text = reasoning_match.group(1).strip() if reasoning_match else ""
                variants.append({"reasoning": reasoning_text, "variant": variant_expr})
    return variants

async def process_single_variant(original_integral: str, difficulty: str, variant_data: dict) -> dict:
    """
    Process one variant dictionary:
      - Attempt to compute its antiderivative (solution) and numerical evaluations.
        If integration takes too long (or otherwise fails) the solution is set to None.
      - Attempt a verification by differentiating the computed solution if available.
      - Ask the LLM for a difficulty evaluation.
    
    Note: Even if the antiderivative computation takes too long, the variant is still returned;
          it just carries a solution of None and verification of None.
    """
    variant_integral = variant_data.get("variant")
    if not variant_integral:
        return None

    # Always attempt to compute the solution; if the integration is too hard (or disabled), solution will be None.
    solution, evaluations = compute_solution_and_evals(variant_integral)

    # Try to verify the computed solution only if available.
    x = sp.symbols('x')
    verification = None
    pattern = r"integrate\((.+),\s*x\)"
    match = re.search(pattern, variant_integral)
    if match:
        try:
            integrand_str = match.group(1)
            integrand = sp.sympify(integrand_str)
            if solution is not None:
                antideriv = sp.sympify(solution)
                diff_expr = sp.simplify(sp.diff(antideriv, x) - integrand)
                verification = (diff_expr == 0)
            else:
                verification = None
        except Exception as e:
            verification = None
    else:
        verification = None

    return {
        "original": original_integral,
        "requested_difficulty": difficulty,
        "variant": variant_integral,
        "reasoning": variant_data.get("reasoning"),
        "variant_response": None,
        "verification_passed": verification,
        "evaluation": None,
        "transformations_used": variant_data.get("transformations_used", []),
        "solution": solution,           # Will be None if integration took too long or if CALCULATE_SYMBOLIC is False.
        "evaluations": evaluations,       # Will be {} if integration took too long or if CALCULATE_SYMBOLIC is False.
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

def get_random_prompt_template(integral_str: str, difficulty: str, count: int, transforms_text: str, personas_str: str) -> str:
    templates = []
    templates.append(
        f"Assume you can adopt various mathematical personas such as {personas_str}.\n\n"
        f"Given the integral: {integral_str}\n"
        f"Your task is to generate {count} creative and unexpected variant(s) that are {difficulty} than the original.\n\n"
        "Follow these steps:\n"
        "1. Analyze the original integral deeply, looking for hidden patterns and non-obvious properties.\n"
        "2. Think outside conventional approaches - consider unusual substitutions, creative identities, or surprising transformations.\n"
        f"3. Draw inspiration from various mathematical fields. Some ideas: {transforms_text}\n"
        "4. Provide a detailed explanation of your creative reasoning process.\n"
        "5. Present each variant in valid Python sympy syntax in the form: integrate(<integrand>, x).\n\n"
        "Push yourself to find truly novel variants that might surprise even experienced mathematicians!\n\n"
        "Return your answer in the following exact format for each variant:\n"
        "====\n"
        "Variant <number>:\n"
        "Reasoning: <your creative chain-of-thought explanation>\n"
        "Variant: integrate(<integrand>, x)\n"
        "===="
    )
    
    # Add another template that emphasizes different creative aspects
    templates.append(
        f"Channel the creative spirit of great mathematicians like {personas_str}.\n\n"
        f"For this integral: {integral_str}\n"
        f"Create {count} mathematically interesting variant(s) that are {difficulty} than the original.\n\n"
        "Steps:\n"
        "1. Look for hidden mathematical beauty and unexpected connections in the original integral.\n"
        "2. Consider how different areas of mathematics might offer surprising approaches.\n"
        f"3. Experiment with transformations such as {transforms_text}\n"
        "4. Explain your mathematical insights and creative process.\n"
        "5. Express each variant using valid Python sympy syntax: integrate(<integrand>, x)\n\n"
        "Aim to create variants that showcase the rich interconnections in mathematics!\n\n"
        "Use this format:\n"
        "====\n"
        "Variant <number>:\n"
        "Reasoning: <your creative chain-of-thought explanation>\n"
        "Variant: integrate(<integrand>, x)\n"
        "===="
    )
    
    return random.choice(templates)

async def generate_variant_chunk(integral_str: str, difficulty: str, count: int) -> list:
    """
    Generate a chunk (up to 10) of variants in a single LLM call.
    The prompt instructs the LLM to produce `count` variants in the specified format.
    After receiving the response, each variant is parsed and then further processed.
    """
    # Get a list of transformation ideas based on difficulty.
    transformations = TRANSFORMATIONS_BY_DIFFICULTY.get(difficulty.lower(), [])
    if not transformations:
        transformations = ["make a small change"]

    # Pick a random subset (between 3 and 6) from the transformation list.
    num_choices = random.choice(range(3, 7))
    chosen_transforms = random.sample(transformations, min(num_choices, len(transformations)))
    transforms_text = ", ".join(chosen_transforms)

    # Expand the persona list for more diverse creative inspiration.
    personas = [
        "Richard Feynman who loves finding intuitive physical interpretations",
        "Leonhard Euler who excels at infinite series and creative substitutions",
        "Carl Friedrich Gauss who sees deep mathematical patterns",
        "Emmy Noether who focuses on symmetry and invariance",
        "Paul Dirac who prefers elegant mathematical beauty",
        "Isaac Newton who thinks in terms of physical motion and rates of change",
        "Gottfried Leibniz who seeks systematic notation and patterns",
        "Bernhard Riemann who explores complex geometric relationships",
        "Pierre-Simon Laplace who excels at transform methods",
        "Joseph-Louis Lagrange who loves analytical mechanics approaches",
        "Henri Poincaré who sees topological connections",
        "Srinivasa Ramanujan who has incredible intuition for identities",
        "David Hilbert who approaches problems with rigorous formalism",
        "John von Neumann who combines computational and theoretical insights",
        "Sophie Germain who finds innovative prime number relationships",
        "George Pólya who uses creative problem-solving strategies",
        "Augustin-Louis Cauchy who emphasizes rigorous analysis",
        "Évariste Galois who sees algebraic structure in everything",
        "Ada Lovelace who thinks algorithmically",
        "Alan Turing who approaches problems computationally",
        "Kurt Gödel who seeks logical foundations",
        "Edward Witten who applies physics insights to mathematics",
        "Terence Tao who combines multiple mathematical disciplines",
        "Katherine Johnson who excels at practical numerical computations",
        "Maryam Mirzakhani who thinks in terms of geometric dynamics",
        "a calculus professor who loves elegant simplifications",
        "a creative mathematician who enjoys unusual substitutions",
        "a student who prefers working with polynomials and rational functions",
        "a theoretical physicist who likes trigonometric and exponential forms",
        "an engineer who favors practical, computational approaches",
        "a number theorist fascinated by prime numbers and rational coefficients",
        "a geometry enthusiast who thinks in terms of geometric transformations",
        "an algebraic geometer with a penchant for symmetry",
        "a computational mathematician who values algorithmic efficiency",
        "Peter Gustav Lejeune Dirichlet who masters conditional convergence",
        "Carl Gustav Jacob Jacobi who specializes in elliptic functions",
        "William Rowan Hamilton who thinks in quaternions",
        "Sofia Kovalevskaya who masters partial differential equations",
        "Hermann Weyl who combines geometry with group theory",
        "André Weil who sees algebraic geometry everywhere",
        "Paul Erdős who finds elementary yet deep approaches",
        "Benoit Mandelbrot who thinks in fractals and self-similarity",
        "Stephen Hawking who applies cosmological intuition",
        "Hermann Minkowski who thinks in spacetime geometry",
        "Felix Klein who sees geometric symmetries"
    ]
    personas_str = ", ".join(personas)

    # Randomly choose a prompt template.
    prompt_variant = get_random_prompt_template(integral_str, difficulty, count, transforms_text, personas_str)

    # Randomize temperature for creativity.
    temperature_choice = random.choice([0.8, 1.0, 1.2, 1.4])
    response_text = await generate_text(MODEL, prompt_variant, temperature=temperature_choice)
    parsed_variants = parse_variants(response_text)

    for variant in parsed_variants:
        variant["transformations_used"] = chosen_transforms

    tasks = [
        process_single_variant(integral_str, difficulty, variant)
        for variant in parsed_variants
    ]
    processed_variants = await asyncio.gather(*tasks)
    # Filter out any None results (if any parsing issues occur)
    return [v for v in processed_variants if v is not None]

# Expanded transformation ideas for extra variance.
TRANSFORMATIONS_BY_DIFFICULTY = {
    "easier": [
        "remove a complicated term",
        "simplify the denominator",
        "reduce an exponent",
        "change a function to an easier one",
        "lower a coefficient",
        "remove a factor",
        "eliminate a radical",
        "drop a subexpression",
        "simplify a trigonometric component",
        "convert a product to a simpler sum",
        "replace a complex fraction with a simpler one",
        "remove nested functions",
        "reduce the degree of a polynomial",
        "simplify composite trigonometric functions",
        "remove logarithmic terms",
        "eliminate absolute value terms",
        "reduce the number of terms in the expression",
        "replace transcendental functions with simpler algebraic ones",
        "factor common elements",
        "cancel redundant terms",
        "linearize the integrand",
        "break up a compound fraction",
        "simplify rational expressions",
        "remove redundant constants",
        "combine like terms to reduce complexity",
        "split a complex fraction into partial fractions",
        "substitute a simpler function using known derivatives",
        "remove a squared term from denominator",
        "convert exponential to simpler polynomial",
        "replace hyperbolic functions with exponentials",
        "substitute a known derivative pattern",
        "remove cross terms in denominator",
        "reduce number of distinct variables",
        "convert to a recognizable standard form",
        "simplify using common factor extraction",
        "remove nested square roots",
        "convert to a single fraction",
        "eliminate double angles in trigonometric terms",
        "reduce to basic trigonometric ratios",
        "simplify using completion of square",
        "convert to partial fractions with simpler terms",
        "remove complex conjugates",
        "reduce to elementary functions",
        "simplify using function composition",
        "convert to a recognizable derivative form",
        "reduce to basic algebraic operations",
        "simplify using known antiderivative patterns",
        "convert to a standard integration form"
    ],
    "equivalent": [
        # Creative structural transformations
        "rewrite using a clever substitution that maintains difficulty",
        "apply a creative trigonometric or hyperbolic identity",
        "use an unexpected algebraic manipulation",
        "introduce an auxiliary function that cancels out",
        "employ a surprising symmetry or pattern",
        "use a non-obvious completion of square",
        "apply a creative partial fraction decomposition",
        "introduce conjugates in an interesting way",
        "transform using a clever reciprocal relationship",
        "rewrite using an unexpected function composition",
        
        # Advanced mathematical techniques
        "employ residue theory concepts",
        "use contour integration ideas",
        "apply Fourier transform thinking",
        "utilize complex analysis viewpoints",
        "think in terms of differential equations",
        "consider series expansion perspectives",
        "apply group theory symmetries",
        "use geometric transformation insights",
        
        # Keep existing transformations
        "change a function to a different but equivalent one",
        "change coefficient values slightly",
        "alter constant terms",
        "modify an exponent slightly",
        "rewrite the integrand in a different form without changing overall complexity",
        "exchange similar functions (e.g., sin to cos)",
        "adjust parameters while keeping the integral equivalent",
        "rearrange the order of terms",
        "use trigonometric identities to rewrite the expression",
        "substitute equivalent exponential forms",
        "change variables while maintaining complexity",
        "distribute terms differently",
        "factor common terms in a new way",
        "rewrite using alternate algebraic forms",
        "swap numerator and denominator with reciprocal",
        "use alternate but equivalent radical forms",
        "rewrite using different logarithmic properties",
        "apply integration by substitution with a trivial substitution",
        "apply partial fractions in an equivalent manner",
        "rationalize the integrand slightly"
    ],
    "harder": [
        "introduce an additional polynomial factor",
        "increase the exponent",
        "add a non-linear term",
        "include a higher degree term",
        "insert a logarithmic factor",
        "complicate the denominator",
        "introduce a composite trigonometric function",
        "add a product of functions",
        "embed an extra constant factor that makes the expression less trivial",
        "introduce a nested function",
        "compose with a non-linear transformation",
        "incorporate a trigonometric identity in reverse",
        "insert a non-trivial logarithmic composition",
        "add an oscillatory term",
        "mix in a transcendental function",
        "introduce a fractional exponent",
        "include a hyperbolic function element",
        "add an extra variable substitution step",
        "incorporate a rational function with higher degree",
        "complicate with a piecewise component"
    ]
}

async def process_integral(integral_str: str, difficulties: list, num_variants: int = 3) -> list:
    """
    Generate a batch of variants for the given integral and for each difficulty.
    If more than 10 variants are requested per difficulty, the work is split into multiple LLM calls.
    A buffer multiplier is used to allow for duplicates or filtering before trimming to the requested number.
    """
    final_results = []
    seen_variants = set()
    buffer_multiplier = 3  # increased to generate more candidate variants
    tasks = []

    for difficulty in difficulties:
        total_to_request = num_variants * buffer_multiplier
        num_chunks = math.ceil(total_to_request / 10)
        for i in range(num_chunks):
            count = 10 if (i < num_chunks - 1) else (total_to_request - 10 * (num_chunks - 1))
            tasks.append((difficulty, generate_variant_chunk(integral_str, difficulty, count)))

    chunk_results = await asyncio.gather(*[t[1] for t in tasks])
    difficulty_dict = {d: [] for d in difficulties}
    for idx, (difficulty, _) in enumerate(tasks):
        for variant in chunk_results[idx]:
            variant_expr = variant.get("variant")
            # Do not discard the variant based on integration difficulty; only filter based on LLM's evaluation.
            if (variant_expr 
                and variant_expr not in seen_variants 
                and not (variant.get("evaluation", "") == "harder" and difficulty != "harder")):
                seen_variants.add(variant_expr)
                difficulty_dict[difficulty].append(variant)
    
    for difficulty in difficulties:
        final_results.extend(difficulty_dict[difficulty][:num_variants])
    
    return final_results

async def main():
    base_integral = "integrate(1/(x**2 - x + 1), x)"
    difficulties = ["easier", "equivalent", "harder"]
    print("Processing integral:", base_integral)
    variants = await process_integral(base_integral, difficulties, num_variants=3)
    
    with open("variants.json", "w") as outfile:
        json.dump(variants, outfile, indent=2)
    
    for idx, v in enumerate(variants, start=1):
        print(f"\n--- Variant {idx} ---")
        print("Requested difficulty:", v["requested_difficulty"])
        print("Transformations used:", v["transformations_used"])
        print("Variant integral:", v["variant"])
        print("Verification passed:", v["verification_passed"])
        print("LLM evaluation:", v["evaluation"])
        print("Solution (antiderivative):", v["solution"])
        print("Evaluations at random points:", v["evaluations"])

if __name__ == "__main__":
    asyncio.run(main())

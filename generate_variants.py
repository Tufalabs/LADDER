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

from utils.inference import generate_text

# Global process pool executor.
executor = concurrent.futures.ProcessPoolExecutor(max_workers=4)

def integrate_wrapper(integrand, x):
    """Compute sp.integrate(integrand, x) for indefinite integrals."""
    return sp.integrate(integrand, x)

def run_integration(integrand, x, timeout=TIMEOUT_SECONDS):
    """Run indefinite integration with a timeout."""
    future = executor.submit(integrate_wrapper, integrand, x)
    return future.result(timeout=timeout)

def definite_integrate_wrapper(integrand, x, lower, upper):
    """Compute sp.integrate(integrand, (x, lower, upper)) for definite integrals."""
    return sp.integrate(integrand, (x, lower, upper))

def run_definite_integration(integrand, x, lower, upper, timeout=TIMEOUT_SECONDS):
    """Run definite integration with a timeout."""
    future = executor.submit(definite_integrate_wrapper, integrand, x, lower, upper)
    return future.result(timeout=timeout)

def is_definite_integral(integral_str: str) -> bool:
    """
    Return True if the integral string contains definite limits of the form (x, a, b).
    """
    pattern = r"integrate\([^,]+,\s*\(x,\s*[^,]+,\s*[^)]+\)\)"
    return re.search(pattern, integral_str) is not None

def verify_integral(integral_str: str) -> bool:
    """
    Verify the antiderivative by checking that differentiating it gives back the integrand.
    (For definite integrals this check is skipped.)
    """
    x = sp.symbols('x')
    if is_definite_integral(integral_str):
        # For definite integrals, we assume the computed result is correct.
        return True

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
    Compute the antiderivative for an indefinite integral or the definite integral result
    if limits are provided. For definite integrals, the computed result is returned as the solution,
    and evaluations is a dictionary with the numerical result.
    """
    if not CALCULATE_SYMBOLIC:
        return None, {}

    x = sp.symbols('x')
    if is_definite_integral(integral_str):
        # Process as a definite integral.
        pattern = r"integrate\((.+),\s*\(x,\s*([^,]+),\s*([^)]+)\)\)"
        match = re.search(pattern, integral_str)
        if not match:
            return None, {}

        integrand_str = match.group(1)
        lower_limit_str = match.group(2)
        upper_limit_str = match.group(3)
        try:
            integrand = sp.sympify(integrand_str)
            lower_limit = sp.sympify(lower_limit_str)
            upper_limit = sp.sympify(upper_limit_str)
        except Exception as e:
            print("Error sympifying definite integral parts:", e)
            return None, {}

        try:
            result = run_definite_integration(integrand, x, lower_limit, upper_limit, timeout=TIMEOUT_SECONDS)
        except Exception as e:
            print("Integration timed out in compute_solution_and_evals (definite); marking as too hard.")
            return None, {}

        solution_str = str(result)
        try:
            result_eval = float(result.evalf())
        except Exception as e:
            result_eval = None
        evaluations = {"definite_result": result_eval}
        return solution_str, evaluations
    else:
        # Process as an indefinite integral.
        pattern = r"integrate\((.+),\s*x\)"
        match = re.search(pattern, integral_str)
        if not match:
            return None, {}

        integrand_str = match.group(1)
        try:
            integrand = sp.sympify(integrand_str)
        except Exception as e:
            print("Error sympifying integrand:", e)
            return None, {}

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

def parse_variants(text: str) -> list:
    """
    Parse the LLM response text and extract a list of variant dictionaries.
    Each variant is expected to be delimited by "====" and include both a Reasoning and a Variant line.
    This version extracts the entire line following 'Variant:'.
    """
    variants = []
    blocks = re.split(r"====\s*", text)
    for block in blocks:
        if "Variant:" in block and "Reasoning:" in block:
            # Extract reasoning using regex
            reasoning_match = re.search(r"Reasoning:\s*(.*?)\s*Variant:", block, re.DOTALL)
            reasoning_text = reasoning_match.group(1).strip() if reasoning_match else ""
            
            # Split block into lines and find the line that starts with "Variant:"
            variant_expr = None
            for line in block.splitlines():
                if line.strip().startswith("Variant:"):
                    # Remove the 'Variant:' prefix and any extra whitespace.
                    variant_expr = line.strip()[len("Variant:"):].strip()
                    break
            # Optional: if the variant expression doesn't end with a ')', skip it
            if variant_expr and variant_expr.endswith(")"):
                variants.append({"reasoning": reasoning_text, "variant": variant_expr})
    return variants


async def process_single_variant(original_integral: str, difficulty: str, variant_data: dict) -> dict:
    """
    Process one variant dictionary:
      - Compute its solution (either antiderivative or definite result) and evaluations.
      - For indefinite integrals, verify the computed antiderivative by differentiation.
      - For definite integrals, we assume verification passes.
    """
    variant_integral = variant_data.get("variant")
    if not variant_integral:
        return None

    solution, evaluations = compute_solution_and_evals(variant_integral)
    x = sp.symbols('x')
    definite_pattern = r"integrate\((.+),\s*\(x,\s*([^,]+),\s*([^)]+)\)\)"
    if re.search(definite_pattern, variant_integral):
        verification = True
    else:
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
        "solution": solution,
        "evaluations": evaluations,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

def get_random_prompt_template(integral_str: str, difficulty: str, count: int, transforms_text: str, personas_str: str) -> str:
    """
    Return a prompt template that instructs the LLM to produce variants.
    If the base integral is definite then require variants in the form
      integrate(<integrand>, (x, lower_limit, upper_limit))
    otherwise use the indefinite form.
    """
    if is_definite_integral(integral_str):
        expected_format = "integrate(<integrand>, (x, lower_limit, upper_limit))"
    else:
        expected_format = "integrate(<integrand>, x)"
    
    templates = []
    templates.append(
        f"Assume you can adopt various mathematical personas such as {personas_str}.\n\n"
        f"Given the integral: {integral_str}\n"
        f"Your task is to generate {count} creative and unexpected variant(s) that are {difficulty} than the original.\n\n"
        "Important constraints:\n"
        "- Always use 'x' as the variable of integration\n"
        "- Do not introduce arbitrary constants - all constants must have specific numerical values\n\n"
        "Follow these steps:\n"
        "1. Analyze the original integral deeply, looking for hidden patterns and non-obvious properties.\n"
        "2. Think outside conventional approaches – consider unusual substitutions, creative identities, or surprising transformations.\n"
        f"3. Draw inspiration from various mathematical fields. Some ideas: {transforms_text}\n"
        "4. Provide a detailed explanation of your creative reasoning process.\n"
        f"5. Present each variant in valid Python sympy syntax in the form: {expected_format}.\n\n"
        "Push yourself to find truly novel variants that might surprise even experienced mathematicians!\n\n"
        "Return your answer in the following exact format for each variant:\n"
        "====\n"
        "Variant <number>:\n"
        "Reasoning: <your creative chain-of-thought explanation>\n"
        f"Variant: {expected_format}\n"
        "===="
    )
    
    templates.append(
        f"Channel the creative spirit of great mathematicians like {personas_str}.\n\n"
        f"For this integral: {integral_str}\n"
        f"Create {count} mathematically interesting variant(s) that are {difficulty} than the original.\n\n"
        "Critical requirements:\n"
        "- Use 'x' as the integration variable - do not change this\n"
        "- All constants must be specific numbers (e.g., 2, 3.14, etc.) - no arbitrary constants\n\n"
        "Steps:\n"
        "1. Look for hidden mathematical beauty and unexpected connections in the original integral.\n"
        "2. Consider how different areas of mathematics might offer surprising approaches.\n"
        f"3. Experiment with transformations such as {transforms_text}\n"
        "4. Explain your mathematical insights and creative process.\n"
        f"5. Express each variant using valid Python sympy syntax: {expected_format}\n\n"
        "Aim to create variants that showcase the rich interconnections in mathematics!\n\n"
        "Use this format:\n"
        "====\n"
        "Variant <number>:\n"
        "Reasoning: <your creative chain-of-thought explanation>\n"
        f"Variant: {expected_format}\n"
        "===="
    )
    
    return random.choice(templates)

async def generate_variant_chunk(integral_str: str, difficulty: str, count: int) -> list:
    """
    Generate a chunk (up to 10) of variants in a single LLM call.
    The prompt instructs the LLM to produce `count` variants in the specified format.
    After receiving the response, each variant is parsed and further processed.
    """
    transformations = TRANSFORMATIONS_BY_DIFFICULTY.get(difficulty.lower(), [])
    if not transformations:
        transformations = ["make a small change"]

    num_choices = random.choice(range(3, 7))
    chosen_transforms = random.sample(transformations, min(num_choices, len(transformations)))
    transforms_text = ", ".join(chosen_transforms)

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

    prompt_variant = get_random_prompt_template(integral_str, difficulty, count, transforms_text, personas_str)
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
    return [v for v in processed_variants if v is not None]

TRANSFORMATIONS_BY_DIFFICULTY = {
    "easier": [
        "simplify the denominator",
        "reduce exponents",
        "remove complex terms",
        "convert to partial fractions",
        "simplify trigonometric terms",
        "combine like terms",
        "factor common elements",
        "remove nested functions",
        "convert to standard forms",
        "linearize the integrand",
        "eliminate radicals",
        "reduce polynomial degree",
        "split compound fractions",
        "remove logarithmic terms",
        "substitute simpler functions"
    ],
    "equivalent": [
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
        "increase complexity of terms",
        "add additional factors",
        "introduce higher degrees",
        "combine multiple functions",
        "add nested functions",
        "use function composition",
        "incorporate transcendental functions",
        "mix different function types",
        "increase algebraic complexity",
        "add non-elementary functions",
        "use more complex substitutions",
        "increase number of terms",
        "add rational complexity",
        "combine different mathematical concepts",
        "increase structural complexity"
    ]
}

async def process_integral(integral_str: str, difficulties: list, num_variants: int = 3) -> list:
    """
    Generate a batch of variants for the given integral and for each difficulty.
    If more than 10 variants are requested per difficulty, the work is split into multiple LLM calls.
    """
    final_results = []
    seen_variants = set()
    buffer_multiplier = 3
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
            if (variant_expr 
                and variant_expr not in seen_variants 
                and not (variant.get("evaluation", "") == "harder" and difficulty != "harder")):
                seen_variants.add(variant_expr)
                difficulty_dict[difficulty].append(variant)
    
    for difficulty in difficulties:
        final_results.extend(difficulty_dict[difficulty][:num_variants])
   
    return final_results

async def main():
    base_integral = "integrate(1/(x**2 - x + 1), (x, 0, 1))"
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
        print("Solution (integral result):", v["solution"])
        print("Evaluations:", v["evaluations"])

if __name__ == "__main__":
    asyncio.run(main())

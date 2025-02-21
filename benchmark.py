"""
eval_intergrals.py

This script evaluates whether an LLM-generated solution for an integral is correct.
It:
  1. Prompts the LLM to solve an integral (provided as text) and return the solution 
     wrapped in <answer> ... </answer> tags as a valid Python sympy expression.
     For definite integrals (with bounds specified as "from x=... to x=..."), the answer should be the numerical value.
     For indefinite integrals, the answer should be an antiderivative.
  2. Extracts the candidate solution from within the <answer> tags while retaining the full raw LLM response.
  3. Verifies the candidate solution using numerical methods:
       - For definite integrals, it compares the candidate value with a numerical quadrature result.
       - For indefinite integrals, it evaluates F(b)-F(a) on random intervals and compares with numerical integration.
  4. Returns the candidate solution, the computed solution (or numerical result), the full raw LLM response, and a correctness flag.
  
Additionally, this script loads a list of integrals from a questions file, benchmarks the model's performance
across many integrals, prints progress and an overall accuracy summary, and optionally saves a new Python file
with only the integrals that were answered incorrectly.

Usage example:
    python eval_intergrals.py
"""

import re
import random
import sympy as sp
import asyncio
import mpmath as mp  # for numerical integration
from utils.inference import generate_text  # Ensure this function is available
from questions.mit_bee_qualifiying_questions_2025test import BASE_QUESTIONS as QUESTIONS

# ---------------- Configuration ----------------
SAVE_INCORRECT_QUESTIONS = True  # Set to True to save a new file with the integrals answered incorrectly.
INCORRECT_FILENAME = "incorrect_hardest_questions.py"  # Name of the file to save incorrect questions.

# ---------------- Helper Functions ----------------

def preprocess_integral_text(text: str) -> str:
    """
    Preprocesses the integral text to help Sympy parse advanced constructs.
    Also converts standard mathematical notation to our format.
    """
    text = text.replace("sum(", "Sum(")
    text = text.replace("ContinuedFraction(", "continued_fraction(")
    
    # Convert standard mathematical notation to our format
    pattern = r"integrate\((.*?),\s*x,\s*(.*?),\s*(.*?)\)"
    match = re.match(pattern, text)
    if match:
        integrand, lower, upper = match.groups()
        text = f"integrate({integrand}, x) from x={lower} to x={upper}"
    
    return text

def extract_candidate_solution(text: str) -> str:
    """
    Extracts a candidate solution from the LLM response by looking for text wrapped in <answer> and </answer> tags
    or [box] and [/box] tags.
    
    If these tags are not found, the entire text is returned.
    """
    match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    
    match = re.search(r"\[box\](.*?)\[/box\]", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return text.strip()

def preprocess_candidate_solution(solution: str) -> str:
    """
    Pre-processes the candidate solution string to convert common LaTeX-style expressions
    to Sympy-friendly syntax.
    
    This function:
      - Removes LaTeX math delimiters such as "\(" and "\)", as well as "$" signs.
      - Replaces "\arctan" with "atan" (and you can add more replacements as needed).
      - Replaces "\ln" with "log".
      - Removes any trailing "+ C" or constant additions.
    """
    # Remove LaTeX delimiters
    solution = solution.replace(r"\(", "").replace(r"\)", "")
    solution = solution.replace("$", "")
    # Replace LaTeX-style commands with sympy equivalents
    solution = solution.replace("\\arctan", "atan")
    solution = solution.replace("\\ln", "log")
    # Remove any occurrence of "+ C" or " + constant"
    solution = re.sub(r"\+?\s*C", "", solution)
    return solution.strip()

# Define a locals dictionary to help Sympy parse functions and constants.
LOCALS_DICT = {
    "pi": sp.pi,
    "oo": sp.oo,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "cot": sp.cot,
    "csc": sp.csc,
    "sec": sp.sec,
    "log": sp.log,
    "exp": sp.exp,
    "factorial": sp.factorial,
    "Sum": sp.Sum,
    "sum": sp.Sum,  # map lowercase 'sum' to sympy.Sum
    "arctan": sp.atan,
    "arctanh": sp.atanh,
    "arccos": sp.acos,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "continued_fraction": sp.continued_fraction,
}

async def evaluate_llm_solution(integral_text: str, num_tests: int = 10, model: str = "gpt-4o-mini", tol: float = 1e-2):
    """
    Uses the LLM to generate a solution for the given integral and then verifies its correctness using numerical methods.
    
    This function supports both definite and indefinite integrals.
    
    For definite integrals (with bounds specified as 'from x=<lower> to x=<upper>'):
        - The candidate solution is expected to be the value of the definite integral.
        - Verification is done by comparing the candidate value with mp.quad numerical integration.
    
    For indefinite integrals:
        - The candidate solution is expected to be an antiderivative.
        - Verification is done by evaluating F(b)-F(a) for random intervals and comparing with mp.quad.
    
    Parameters:
        integral_text: A string of the form "integrate(<integrand>, x)" or 
                       "integrate(<integrand>, x) from x=<lower> to x=<upper>"
        num_tests: Number of random test intervals for verification (only used for indefinite integrals).
        tol: Tolerance for numerical verification.
    
    Returns:
        A tuple (candidate_solution, computed_solution, raw_llm_solution, is_correct) where:
            - candidate_solution: The solution extracted from the LLM's output after preprocessing.
            - computed_solution: The computed solution (or numerical result).
            - raw_llm_solution: The full raw response returned by the LLM.
            - is_correct: True if the candidate solution appears to be correct.
    """
    # Preprocess the integral text to handle advanced constructs.
    integral_text = preprocess_integral_text(integral_text)
    x = sp.symbols('x')
    
    # Updated regex to extract the integrand and optional bounds.
    pattern = r"integrate\((.+),\s*x\)(?:\s*from\s*x\s*=\s*(.+?)\s*to\s*x\s*=\s*(.+))?"
    m = re.search(pattern, integral_text)
    if not m:
        raise ValueError("Could not extract the integrand from the provided integral text.")
    
    integrand_str = m.group(1)
    lower_bound_str = m.group(2)
    upper_bound_str = m.group(3)
    
    try:
        integrand_expr = sp.sympify(integrand_str, locals=LOCALS_DICT)
    except Exception as e:
        return "", "Error parsing integrand: " + str(e), "", False
    
    # Check if bounds are provided (i.e. a definite integral).
    definite = False
    if lower_bound_str is not None and upper_bound_str is not None:
        definite = True
        try:
            lower_bound = sp.sympify(lower_bound_str, locals=LOCALS_DICT)
            upper_bound = sp.sympify(upper_bound_str, locals=LOCALS_DICT)
        except Exception as e:
            return "", "Error parsing bounds: " + str(e), "", False
    
    # Prepare the prompt based on whether the integral is definite or indefinite.
    if definite:
        # Compute the definite integral numerically using mpmath.
        integrand_func = sp.lambdify(x, integrand_expr, "mpmath")
        a_val = float(lower_bound.evalf())
        b_val = float(upper_bound.evalf())
        computed_value = mp.quad(integrand_func, [a_val, b_val])
        computed_solution = str(computed_value)
        prompt = (
            f"Solve the following definite integral and return ONLY your answer wrapped in <answer> and </answer> tags.\n"
            f"Output your answer as a valid Python sympy expression. Use 'pi' for π if needed.\n\n"
            f"{integral_text}\n\n"
            "Show your full working out before solving"
        )
    else:
        computed_solution = "N/A (numerical verification used for antiderivative)"
        prompt = (
            f"Solve the following integral and return ONLY your answer wrapped in <answer> and </answer> tags.\n"
            f"Output your answer as a valid Python sympy expression without LaTeX formatting (for example, use 'atan(x)' not '\\arctan(x)').\n\n"
            f"{integral_text}\n\n"
            "Show your full working out before solving, don't include any constants of integration."
            "Example format is <answer> x**2 + x</answer> or <answer> atan(x**2 + x)</answer>"
        )
    
    # Get the full LLM response.
    raw_llm_solution = await generate_text(model, prompt)
    
    # Extract and preprocess the candidate solution.
    candidate_solution = extract_candidate_solution(raw_llm_solution)
    candidate_solution = preprocess_candidate_solution(candidate_solution)
    
    try:
        
        candidate_expr = sp.sympify(candidate_solution, locals=LOCALS_DICT)
    except Exception as e:
        # If candidate parsing fails, return false.
        return candidate_solution, computed_solution, raw_llm_solution, False
    
    # Verification step:
    if definite:
        # For definite integrals, evaluate the candidate expression and compare with numerical integration.
        try:
            candidate_value = float(candidate_expr.evalf())
        except Exception as e:
            return candidate_solution, computed_solution, raw_llm_solution, False
        is_correct = abs(candidate_value - float(computed_value)) < tol
    else:
        # For indefinite integrals, verify by comparing F(b)-F(a) with the numerical definite integral.
        candidate_func = sp.lambdify(x, candidate_expr, "mpmath")
        integrand_func = sp.lambdify(x, integrand_expr, "mpmath")
        is_correct = True
        for _ in range(num_tests):
            a_val = random.uniform(-10, 10)
            b_val = random.uniform(-10, 10)
            if abs(b_val - a_val) < 1e-3:
                continue
            try:
                candidate_diff = candidate_func(b_val) - candidate_func(a_val)
                definite_integral = mp.quad(integrand_func, [a_val, b_val])
            except Exception as conv_e:
                # Check for mpf creation error specifically
                if "cannot create mpf from" in str(conv_e):
                    is_correct = False
                    break
                continue
            if abs(candidate_diff - definite_integral) > tol:
                is_correct = False
                break

    return candidate_solution, computed_solution, raw_llm_solution, is_correct

# -------------- Benchmarking Code --------------

async def benchmark_integrals(batch_size: int = 10, model: str = "gpt-4o-mini"):
    total = len(QUESTIONS)
    correct_count = 0
    results = []
    print(f"Starting benchmark on {total} integrals in batches of {batch_size}...\n")

    # Process the integrals in batches.
    for batch_start in range(0, total, batch_size):
        batch = QUESTIONS[batch_start:batch_start+batch_size]
        tasks = []
        for integral_text in batch:
            tasks.append(evaluate_llm_solution(integral_text, model=model))
        
        # Await the results concurrently for this batch.
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process and print the results for the current batch.
        for i, res in enumerate(batch_results):
            integral_text = batch[i]
            if isinstance(res, Exception):
                candidate, computed, raw_response, is_correct = ("", "", str(res), False)
            else:
                candidate, computed, raw_response, is_correct = res
            
            if is_correct:
                correct_count += 1
            results.append({
                "integral": integral_text,
                "candidate": candidate,
                "computed": computed,
                "raw_response": raw_response,
                "correct": is_correct
            })
            
            # Active printing for progress.
            print("--------------------------------------------------")
            print(f"Integral: {integral_text}")
            print(f"Candidate Solution: {candidate}")
            print(f"Raw LLM Response: {raw_response}")
            # print(f"Computed Solution: {computed}")
            print(f"LLM Correct: {is_correct}")
            print("--------------------------------------------------\n")
        current_accuracy = (correct_count / (batch_start + len(batch))) * 100
        print(f"Batch {batch_start // batch_size + 1}: Current accuracy = {current_accuracy:.2f}%\n")
    
    overall_accuracy = (correct_count / total) * 100
    print("\n--- Benchmarking Complete ---")
    print(f"Total integrals tested: {total}")
    print(f"Total correct: {correct_count}")
    print(f"Overall accuracy: {overall_accuracy:.2f}%")
    return results

def save_incorrect_questions(results, filename: str):
    """
    Given the benchmarking results, extract the integrals that were answered incorrectly
    and save them to a Python file containing a list variable `QUESTIONS_INCORRECT`.
    """
    incorrect_integrals = [res["integral"] for res in results if not res["correct"]]
    content = (
        "# This file is auto-generated and contains the integrals that were answered incorrectly.\n"
        "QUESTIONS_INCORRECT = [\n"
    )
    for integral in incorrect_integrals:
        content += f"    {repr(integral)},\n"
    content += "]\n"
    
    try:
        with open(filename, "w") as f:
            f.write(content)
        print(f"Incorrect questions saved to {filename}")
    except Exception as e:
        print(f"Failed to write incorrect questions to {filename}: {e}")

# Add this function to save the correct questions
def save_correct_questions(results, filename: str):
    """
    Given the benchmarking results, extract the integrals that were answered correctly
    and save them to a Python file containing a list variable `QUESTIONS_CORRECT`.
    """
    correct_integrals = [res["integral"] for res in results if res["correct"]]
    content = (
        "# This file is auto-generated and contains the integrals that were answered correctly.\n"
        "QUESTIONS_CORRECT = [\n"
    )
    for integral in correct_integrals:
        content += f"    {repr(integral)},\n"
    content += "]\n"
    
    try:
        with open(filename, "w") as f:
            f.write(content)
        print(f"Correct questions saved to {filename}")
    except Exception as e:
        print(f"Failed to write correct questions to {filename}: {e}")

# ---------------- Main Entry Point ----------------

async def test_definite_integral():
    """
    Test function to verify definite integral handling.
    """
    test_cases = [
        # Standard format
        "integrate((2*x - 1 + log(2*x)/log(2)), x, 1, 2)",
        # Original format
        "integrate((2**(x - 1) + log(2*x)/log(2)), x) from x=1 to x=2"
    ]
    
    for test_case in test_cases:
        print(f"\nTesting integral: {test_case}")
        candidate, computed, raw_response, is_correct = await evaluate_llm_solution(test_case)
        print(f"Candidate solution: {candidate}")
        print(f"Computed solution: {computed}")
        print(f"Raw response: {raw_response}")
        print(f"Test result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")

if __name__ == "__main__":
    async def main():
        # # Run the test first
        # print("Running definite integral test...")
        # await test_definite_integral()
        
        print("\nRunning full benchmark...")
        results = await benchmark_integrals(batch_size=100, model="o1-mini")
        if SAVE_INCORRECT_QUESTIONS:
            save_incorrect_questions(results, INCORRECT_FILENAME)
        save_correct_questions(results, "correct_hardest_questions.py")
    
    asyncio.run(main())

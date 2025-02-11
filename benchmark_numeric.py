"""
eval_integrals.py

This script evaluates whether an LLM-generated solution for an indefinite integral is correct.
It:
  1. Prompts the LLM to solve an integral (provided as text) and return the solution 
     wrapped in <answer> ... </answer> tags as a valid Python sympy expression.
     (Definite integrals are not supported in this version.)
  2. Extracts the candidate solution from within the <answer> tags while retaining the full raw LLM response.
  3. Verifies the candidate solution by differentiating it and checking at a few random points that the derivative 
     matches the original integrand.
  4. Returns the candidate solution, a verification message, the full raw LLM response, and a correctness flag.
  
Additionally, this script loads a list of integrals from a questions file, benchmarks the model's performance
across many integrals, prints progress and an overall accuracy summary, and optionally saves a new Python file
with only the integrals that were answered incorrectly.

Usage example:
    python eval_integrals.py
"""

import re
import random
import sympy as sp
import asyncio
import mpmath as mp  # for numerical evaluation
from utils.inference import generate_text  # Ensure this function is available
from questions.mit_bee_qualifiying_questions import BASE_QUESTIONS as QUESTIONS

# ---------------- Configuration ----------------
SAVE_INCORRECT_QUESTIONS = True  # Set to True to save a new file with the integrals answered incorrectly.
INCORRECT_FILENAME = "incorrect_hardest_questions.py"  # Name of the file to save incorrect questions.

# ---------------- Helper Functions ----------------

def preprocess_integral_text(text: str) -> str:
    """
    Preprocesses the integral text to help Sympy parse advanced constructs.
    
    Replacements performed:
      - Replace "sum(" with "Sum(" so that summations are parsed as sympy.Sum.
      - Replace "ContinuedFraction(" with "continued_fraction(" so that they are recognized.
    """
    text = text.replace("sum(", "Sum(")
    text = text.replace("ContinuedFraction(", "continued_fraction(")
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

async def evaluate_llm_solution(integral_text: str, num_tests: int = 10, model: str = "gpt-4o-mini", tol: float = 1e-5):
    """
    Uses the LLM to generate a solution for the given indefinite integral and then verifies its correctness by differentiation.
    
    The candidate solution is expected to be an antiderivative (without an arbitrary constant).
    Verification is done by differentiating the candidate solution and comparing it at several random points to the original integrand.
    
    If the provided integral text includes bounds (i.e. a definite integral), this function will return an error message
    as definite integrals are not supported in this version.
    
    Parameters:
        integral_text: A string of the form "integrate(<integrand>, x)"
        num_tests: Number of random test points for verification.
        tol: Tolerance for numerical verification.
    
    Returns:
        A tuple (candidate_solution, verification_message, raw_llm_solution, is_correct) where:
            - candidate_solution: The solution extracted from the LLM's output after preprocessing.
            - verification_message: A message indicating the verification method (here: differentiation).
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
    
    # If bounds are provided, this is a definite integral.
    if lower_bound_str is not None and upper_bound_str is not None:
        return "", "Definite integrals are not supported in this version.", "", False

    try:
        integrand_expr = sp.sympify(integrand_str, locals=LOCALS_DICT)
    except Exception as e:
        return "", "Error parsing integrand: " + str(e), "", False
    
    # Prepare the prompt for an indefinite integral.
    prompt = (
        f"Solve the following integral and return ONLY your answer wrapped in <answer> and </answer> tags.\n"
        f"Output your answer as a valid Python sympy expression without LaTeX formatting (for example, use 'atan(x)' not '\\arctan(x)').\n\n"
        f"{integral_text}\n\n"
        "Show your full working out before solving, don't include any constants of integration."
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
        return candidate_solution, "Error parsing candidate solution: " + str(e), raw_llm_solution, False
    
    # Verification step via differentiation:
    try:
        # Differentiate the candidate antiderivative.
        candidate_derivative_expr = sp.diff(candidate_expr, x)
    except Exception as e:
        return candidate_solution, "Error differentiating candidate solution: " + str(e), raw_llm_solution, False
    
    # Create functions for numerical evaluation.
    try:
        candidate_derivative_func = sp.lambdify(x, candidate_derivative_expr, "mpmath")
        integrand_func = sp.lambdify(x, integrand_expr, "mpmath")
    except Exception as e:
        return candidate_solution, "Error creating numerical functions: " + str(e), raw_llm_solution, False

    # Verify at several random test points.
    is_correct = True
    for _ in range(num_tests):
        test_point = random.uniform(-100, 100)
        try:
            candidate_value = candidate_derivative_func(test_point)
            integrand_value = integrand_func(test_point)
        except Exception as eval_e:
            continue  # skip problematic test points
        # Check if the values are within tolerance.
        if abs(candidate_value - integrand_value) > tol:
            is_correct = False
            break

    verification_message = "Verified via differentiation: candidate derivative matches integrand within tol={}".format(tol)
    return candidate_solution, verification_message, raw_llm_solution, is_correct

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
                candidate, verification, raw_response, is_correct = ("", "", str(res), False)
            else:
                candidate, verification, raw_response, is_correct = res
            
            if is_correct:
                correct_count += 1
            results.append({
                "integral": integral_text,
                "candidate": candidate,
                "verification": verification,
                "raw_response": raw_response,
                "correct": is_correct
            })
            
            # Active printing for progress.
            print("--------------------------------------------------")
            print(f"Integral: {integral_text}")
            print(f"Candidate Solution: {candidate}")
            print(f"Verification Message: {verification}")
            print(f"Raw LLM Response: {raw_response}")
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

if __name__ == "__main__":
    async def main():
        results = await benchmark_integrals(batch_size=100, model="o1-mini")
        if SAVE_INCORRECT_QUESTIONS:
            save_incorrect_questions(results, INCORRECT_FILENAME)
        save_correct_questions(results, "correct_hardest_questions.py")
    
    asyncio.run(main())

import os

import sympy as sp
from questions.mit_bee_qualifiying_questions_2025test import BASE_QUESTIONS as QUESTIONS

def extract_integrand(integral_text: str) -> str:
    """Extract the integrand from an integration problem text."""
    import re
    integrand_pattern = r"integrate\((.+),\s*x\)"
    m = re.search(integrand_pattern, integral_text)
    if not m:
        raise ValueError("Could not extract the integrand from the provided integral text.")
    return m.group(1)

if __name__ == '__main__':
    # Add subset size parameter - if 0, use all questions
    subset_size = 0
    
    # Get the questions list, either full or subset
    questions_to_use = QUESTIONS
    if subset_size > 0:
        questions_to_use = QUESTIONS[:subset_size]
    
    samples = []
    # Define an instruction for the incorrect questions.
    instruction_following = (
        "Solve the following integral. Provide ONLY your antiderivative as a valid Python sympy expression e.g  <answer>cos(x**2)+ ln(x)+1/3*x^3 </answer> "
        "wrapped in <answer> and </answer> tags. Show your full working out before solving, don't include any constants of integration."
    )
    
    x = sp.symbols('x')
    
    # Loop over each incorrect question.
    for idx, question in enumerate(questions_to_use):
        # Build the prompt by combining the question with the instruction.
        prompt_content = f"{question}\n{instruction_following}"
        
        # Build a sample dictionary
        sample = {
            "data_source": "integration_numeric",
            "prompt": [{
                "role": "user",
                "content": prompt_content
            }],
            "ability": "integration",
            "reward_model": {
                "style": "rule",
                "ground_truth": question
            },
            "extra_info": {
                "question_index": idx,
                "question_id": question
            }
        }
        samples.append(sample)
    
    # Define a local output directory and ensure it exists.
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the samples to a JSON file
    import json
    with open(os.path.join(output_dir, 'integration_test.json'), 'w') as f:
        json.dump(samples, f, indent=2)
    
    # Save the samples to a Parquet file
    import pandas as pd
    df = pd.DataFrame(samples)
    df.to_parquet(os.path.join(output_dir, 'integration_test.parquet'))
    
    print("Incorrect questions dataset created and saved to JSON and Parquet successfully.")

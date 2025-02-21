import os
import json
import pyarrow as pa
import pyarrow.parquet as pq

if __name__ == '__main__':
    # Path to the directories containing your JSON files
    data_dirs = ['variant_results']  # Adjust these paths as needed
    remove_duplicates = True  # Flag to control duplicate removal

    train_samples = []
    seen_integrals = set() if remove_duplicates else None
    duplicates_found = 0
    
    # Define the integration-specific instruction.
    instruction_following = (
        "Solve the following integral. Provide ONLY your antiderivative as a valid Python sympy expression e.g  <answer>cos(x**2)+ ln(x)+1/3*x^3 </answer> "
        "wrapped in <answer> and </answer> tags. Show your full working out before solving, don't include any constants of integration."
    )
    
    total_questions = 0  # Initialize a counter for the total number of questions

    # Loop over every directory in the data_dirs list
    for data_dir in data_dirs:
        # Loop over every JSON file in the specified directory.
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(data_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"Skipping file {filename} due to error: {e}")
                    continue

                # Ensure data is a list. If not, skip the file.
                if not isinstance(data, list):
                    print(f"Skipping file {filename} because its contents are not a list.")
                    continue

                for item in data:
                    # The JSON file might contain a dictionary or a list of dictionaries.
                    if isinstance(item, dict):
                        variants = [item]
                    elif isinstance(item, list):
                        variants = item
                    else:
                        print(f"Skipping item in file {filename} because it is neither dict nor list: {item}")
                        continue

                    for variant in variants:
                        if not isinstance(variant, dict):
                            print("Skipping variant because it is not a dict.")
                            continue

                        integration_expr = variant.get("variant", "")
                        if not isinstance(integration_expr, str):
                            print("Skipping variant because the 'variant' field is not a string.")
                            continue

                        integration_expr = integration_expr.strip()
                        if not integration_expr:
                            continue  # Skip if there is no variant text.

                        # Skip duplicates only if remove_duplicates is True
                        if remove_duplicates and integration_expr in seen_integrals:
                            duplicates_found += 1
                            continue
                        
                        # Track seen integrals if remove_duplicates is True
                        if remove_duplicates:
                            seen_integrals.add(integration_expr)

                        # Build the prompt by combining the integration expression with the instruction.
                        prompt_content = f"{integration_expr}\n{instruction_following}"
                        
                        # Build a sample dictionary for this variant.
                        sample = {
                            "data_source": "integration_numeric",
                            "prompt": [{
                                "role": "user",
                                "content": prompt_content
                            }],
                            "ability": "integration",
                            "reward_model": {
                                "style": "rule",
                                "ground_truth": integration_expr
                            },
                            "extra_info": {
                                "original": variant.get("original", ""),
                                "requested_difficulty": variant.get("requested_difficulty", ""),
                                "verification_passed": variant.get("verification_passed", False),
                                "reasoning": variant.get("reasoning", ""),
                                "timestamp": variant.get("timestamp", "")
                            }
                        }
                        
                        # Add the sample to the train set.
                        train_samples.append(sample)
                        
                        total_questions += 1  # Increment the counter for each valid question

    # Define a local directory to save the output files.
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the train dataset as a JSON file.
    with open(os.path.join(output_dir, 'integration_train.json'), 'w') as f:
        json.dump(train_samples, f, indent=2)

    # Save the train dataset as a Parquet file.
    train_table = pa.Table.from_pylist(train_samples)
    pq.write_table(train_table, os.path.join(output_dir, 'integration_train.parquet'))

    print(f"Integration dataset created successfully with {total_questions} questions.")
    if remove_duplicates:
        print(f"Skipped {duplicates_found} duplicate integrals.")
    print(f"Final dataset contains {len(train_samples)} unique integrals.")

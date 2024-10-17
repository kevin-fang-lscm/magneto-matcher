import openai
import pandas as pd
import configparser
import os
import json
import time
from tqdm import tqdm

# Load API configuration from config.ini
config = configparser.ConfigParser()
config_file_path = os.path.join(os.path.dirname(__file__), 'config.ini')

# Check if the file exists
if not os.path.exists(config_file_path):
    raise FileNotFoundError(f"Config file not found: {config_file_path}")

config.read(config_file_path)

# Set your OpenAI API key and model from the config file
openai.api_key = config.get('openai', 'api_key')
model = config.get('openai', 'model')


def clean_response(text):
    """Clean the response to ensure it returns a valid list of values."""
    # Split by new lines and remove empty entries or unwanted characters
    return [item.strip().replace('"', '').replace("'", "") for item in text.split('\n') if item.strip()]


def generate_value_variations(value, variation_type="equivalents", timeout_duration=10):
    if variation_type == "equivalents":
        prompt = (f"Generate a list of two semantically equivalent variations for the value '{value}' "
                  f"as it might appear in biomedical databases or research study data. Include a mix of standard names, "
                  f"common abbreviations, typographical errors, and real-world variations. Ensure some challenging cases "
                  f"like ambiguous, misspelled, or domain-specific terms are present. Return only the unnumbered list, "
                  f"with no additional text.")
    elif variation_type == "acronyms":
        prompt = (f"Generate a list of two possible acronyms or abbreviations for the value '{value}', as it might be used "
                  f"in biomedical or scientific databases. Include common short forms, alternate spellings, and variations that "
                  f"may be found in real-world data integration scenarios. Return only the unnumbered list, with no additional text.")
    elif variation_type == "patterns":
        prompt = (f"Generate a list of two variations of the value '{value}' following common naming patterns "
                  f"(e.g., snake_case, camelCase, or title case), as might appear in biomedical or research study data. "
                  f"Include variations that might result from human error or inconsistencies, such as mixed casing or punctuation issues. "
                  f"Return only the unnumbered list, with no additional text.")
    else:
        raise ValueError("Invalid variation type. Choose from 'equivalents', 'acronyms', or 'patterns'.")

    try:
        start_time = time.time()

        # Call OpenAI API to generate the variations
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": ("You are an expert in biomedical and scientific database design and data integration. "
                                "Your task is to generate realistic and diverse variations of data values that might appear in these contexts. "
                                "Follow these guidelines: 1. Consider real-world scenarios where data may include human errors, "
                                "abbreviations, or domain-specific terms. 2. Include challenging cases such as typos, acronyms, "
                                "ambiguous names, and domain-specific terms. 3. The values should reflect the variety of ways data might "
                                "appear in databases or research study datasets. 9. Do not use formatting characters (e.g., hyphens, brackets) at the start of names. 10 Each name should be on a new line without extra characters.")
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            n=1,
            temperature=0.7,
            timeout=5
        )

        # Check if the response takes more than 10 seconds
        if time.time() - start_time > timeout_duration:
            raise TimeoutError("The request took too long.")

        # Extract and clean the response
        generated_text = response.choices[0].message.content.strip()
        cleaned_values = clean_response(generated_text)

        return cleaned_values

    except TimeoutError as e:
        print(f"Timeout error for value '{value}': {e}")
        return []  # Return an empty list in case of a timeout


def generate_variations_for_dataframe(df, filename='value_variations', num_samples=6, save_incrementally=True):
    """Generate variations for all column values in the DataFrame and save incrementally."""

    domain_total = 0
    variations = {}
    json_output_file_path = os.path.join(os.path.dirname(__file__), filename + '.json')

    # Load previous progress if any, to resume from where left off
    if os.path.exists(json_output_file_path):
        with open(json_output_file_path, 'r') as json_file:
            variations = json.load(json_file)

    for column in tqdm(df.columns, desc="Processing columns"):
        
        if column in variations:
            print(f"Skipping column '{column}' (already processed)")
            continue  # Skip already processed columns
        
        distinct_values = df[column].dropna().unique()
        distinct_count = len(distinct_values)

        if distinct_count == 0:
            continue

        # Sample the 50 most frequent values if the distinct count exceeds num_samples
        if distinct_count > num_samples:
            values_to_consider = distinct_values[:num_samples]
        else:
            values_to_consider = distinct_values

        domain_total += len(values_to_consider)

        column_variations = {}
        for value in tqdm(values_to_consider, desc=f"Processing values in column '{column}'", leave=False):

            if value in ["Unknown", "Not Reported", "Not Allowed To Collect", "None", None]:
                continue

            equivalents = generate_value_variations(value, "equivalents")
            acronyms = generate_value_variations(value, "acronyms")
            patterns = generate_value_variations(value, "patterns")

            column_variations[value] = {
                "Equivalents": equivalents,
                "Acronyms": acronyms,
                "Patterns": patterns
            }

            if save_incrementally:
                # Save incrementally as new variations are generated
                variations[column] = column_variations
                with open(json_output_file_path, 'w') as json_file:
                    json.dump(variations, json_file, indent=4)

    print(f"Total distinct values we generated variations for: {domain_total}")
    print(f"Variations saved to {json_output_file_path}")


def main():
    ROOT = '/Users/pena/Library/CloudStorage/GoogleDrive-em5487@nyu.edu/My Drive/NYU - GDrive/arpah/Schema Matching Benchmarks/'
    path = os.path.join(ROOT, 'gdc', 'target-tables', 'gdc_full.csv')
    # path = os.path.join(ROOT, 'gdc', 'target-tables', 'gdc_sample.csv')

    df = pd.read_csv(path)

    # Sample 10% of the columns
    # sampled_columns = df.sample(frac=0.2, axis=1)
    # df = df[sampled_columns.columns]

    print(df.head())

    generate_variations_for_dataframe(df, filename='gdc_value_variations_cont_full')


if __name__ == "__main__":
    main()

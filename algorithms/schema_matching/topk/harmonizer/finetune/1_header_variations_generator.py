import openai
import pandas as pd
import configparser
import os
import json

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
    """Clean the response to ensure it returns a valid list of column names."""
    # Split by new lines and remove empty entries or unwanted characters
    
    return [item.strip().replace('"', '').replace("'", "") for item in text.split('\n') if item.strip()]


def generate_column_variations(column_name, variation_type="equivalents"):

    if variation_type == "equivalents":
        prompt = f"Generate a list of five semantically equivalent names for the column '{
            column_name}' in a database schema. Return only the unnumbered list, with no additional text."
    elif variation_type == "acronyms":
        prompt = f"Generate a list of five possible acronyms or abbreviations for the column '{
            column_name}' for use in database schema or data integration. Return only the unnumbered list, with no additional text."
    elif variation_type == "patterns":
        prompt = f"Generate a list of five variations of the column name '{
            column_name}' following common naming patterns (e.g., snake_case, camelCase, or title case) used in database schemas. Return only the unnumbered list, with no additional text."
    else:
        raise ValueError(
            "Invalid variation type. Choose from 'equivalents', 'acronyms', or 'patterns'.")

    # Call OpenAI API to generate the variations
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert in database schema design and data integration. Your task is to generate diverse and realistic column names that appear in data integration scenarios. Follow these guidelines: 1. Use common database practices and conventions. 2. Cover a wide range of domains (e.g., finance, healthcare, e-commerce, social media). 3. Include both simple and compound names. 4. Use various naming conventions (e.g., snake_case, camelCase, PascalCase). 5. Add common prefixes and suffixes when appropriate. 6. Generate names for both normalized and denormalized schemas. 7. Occasionally use ambiguous or suboptimal names to reflect real-world scenarios. 8. Do not use formatting characters (e.g., hyphens, brackets) at the start of names. 9. Each name should be on a new line without extra characters. Respond only with the requested list of column names, no explanations or commentary."
            },
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        n=1,
        temperature=0.7
    )

    # Extract and clean the response
    generated_text = response.choices[0].message.content.strip()
    cleaned_names = clean_response(generated_text)

    print(f"Generated {variation_type} alternatives for '{
          column_name}':\n{cleaned_names}")

    return cleaned_names


def generate_variations_for_dataframe(df, filename= 'column_variations'):
    """Generate variations for all column names in the DataFrame."""
    variations = {}

    for column in df.columns:
        # print(f"Generating variations for column: {column}")

        # Generate semantically equivalent names
        equivalents = generate_column_variations(column, "equivalents")

        # Generate acronyms/abbreviations
        acronyms = generate_column_variations(column, "acronyms")

        # Generate common naming pattern variations
        patterns = generate_column_variations(column, "patterns")

        # Combine all variations in a dictionary
        variations[column] = {
            "Equivalents": equivalents,
            "Acronyms": acronyms,
            "Patterns": patterns
        }

    # Convert variations to JSON and save to a file
    json_output_file_path = os.path.join(os.path.dirname(__file__), filename+'.json')
    with open(json_output_file_path, 'w') as json_file:
        json.dump(variations, json_file, indent=4)

    print(f"Variations saved to {json_output_file_path}")

    


def main():
    
    ROOT = '/Users/pena/Library/CloudStorage/GoogleDrive-em5487@nyu.edu/My Drive/NYU - GDrive/arpah/Schema Matching Benchmarks/'
    path = os.path.join(ROOT, 'gdc', 'target-tables','gdc_full.csv')

    df = pd.read_csv(path, nrows=5)

    print(df)

    # Generate variations for all columns
    generate_variations_for_dataframe(df, filename='gdc_column_variations')
    

if __name__ == "__main__":
    main()





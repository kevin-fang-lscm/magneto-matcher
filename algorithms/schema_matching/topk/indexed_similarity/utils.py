import pandas as pd
import hashlib
import re
from thefuzz import fuzz


def hash_dataframe(df: pd.DataFrame) -> str:

    hash_object = hashlib.sha256()

    columns_string = ','.join(df.columns) + '\n'
    hash_object.update(columns_string.encode())

    for row in df.itertuples(index=False, name=None):
        row_string = ','.join(map(str, row)) + '\n'
        hash_object.update(row_string.encode())

    return hash_object.hexdigest()


def hash_string_list(string_list):
    """Generates a unique SHA-256 hash for a list of strings."""
    # Concatenate all strings in the list into a single string
    combined_string = ''.join(string_list)
    # Create a new SHA-256 hash object
    hash_object = hashlib.sha256()
    # Update the hash object with the bytes of the combined string
    hash_object.update(combined_string.encode('utf-8'))
    # Get the hexadecimal digest of the hash
    hashed_string = hash_object.hexdigest()
    return hashed_string


def normalize_scores(topk_entries):
    normalized_topk_entries = []
    max_score = max([score for _, score in topk_entries])
    min_score = min([score for _, score in topk_entries])

    for candidate_col, score in topk_entries:
        normalized_score = (score - min_score) / (max_score - min_score)
        normalized_topk_entries.append((candidate_col, normalized_score))

    return normalized_topk_entries


def normalize_string(s):
    # Convert to lowercase
    s = s.lower()
    # Remove non-alphanumeric characters
    s = re.sub(r'[^a-z0-9]', '', s)
    return s


def are_equal_ignore_case_and_punctuation(str1, str2):
    # Normalize both strings
    normalized_str1 = normalize_string(str1)
    normalized_str2 = normalize_string(str2)
    # Compare normalized strings
    return normalized_str1 == normalized_str2


def are_similar_ignore_case_and_punctuation(str1, str2, threshold=90):
    # Normalize both strings
    normalized_str1 = normalize_string(str1)
    normalized_str2 = normalize_string(str2)

    similarity_ratio_score = fuzz.ratio(normalized_str1, normalized_str2)
    similarity_token_score = fuzz.token_sort_ratio(
        normalized_str1, normalized_str2)

    return similarity_ratio_score >= threshold or similarity_token_score >= threshold


def expand_column_name(column_name):
    if not isinstance(column_name, str):
        raise ValueError("Input must be a string")

    # Replace hyphens and underscores with a space
    column_name = re.sub(r'[-_]+', ' ', column_name)

    # Replace other non-alphanumeric characters with a space
    column_name = re.sub(r'[^\w\s]', ' ', column_name)

    # Split camelCase words by inserting a space before each uppercase letter
    column_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', column_name)

    # Split camelCase with numbers by inserting a space before capital letter following a number
    column_name = re.sub(r'(\d)([A-Z])', r'\1 \2', column_name)

    # Split adjacent numbers and text
    column_name = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', column_name)
    column_name = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', column_name)

    # Split the column name by spaces to get the list of words
    words = [word.lower() for word in column_name.split() if word]

    return words


# TODO use stemming or lemmatization
def is_main_theme_similar(term1, term2):

    stopwords = {'of', 'at', 'in', 'the', 'a', 'an',
                 'and', 'on', 'for', 'with', 'to', 'by'}

    expanded_terms1 = expand_column_name(term1)
    expanded_terms2 = expand_column_name(term2)

    expanded_terms1 = {term.lower() for term in expanded_terms1 if len(
        term) > 1 and term.lower() not in stopwords}
    expanded_terms2 = {term.lower() for term in expanded_terms2 if len(
        term) > 1 and term.lower() not in stopwords}

    common_terms = expanded_terms1.intersection(expanded_terms2)

    return len(common_terms) > 0


# s1 = 'Vital.Status'
# s2 = 'vital_status'
# b = are_equal_ignore_case_and_punctuation(s1, s2)
# print(b)

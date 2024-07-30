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

# s1 = 'Vital.Status'
# s2 = 'vital_status'
# b = are_equal_ignore_case_and_punctuation(s1, s2)
# print(b)

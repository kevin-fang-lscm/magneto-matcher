
from .constants import NULL_REPRESENTATIONS, BINARY_VALUES
import numpy as np
import tqdm
import pandas as pd
import re


def is_null_value(value):
    if isinstance(value, str):
        value = value.lower()
    return value in NULL_REPRESENTATIONS


def is_binary_value(value):
    if isinstance(value, str):
        value = value.lower()
    return value in BINARY_VALUES


def remove_invalid_characters(input_string):
    # Remove any character that is not a letter, digit, or whitespace
    pattern = r'[^a-zA-Z0-9\s]'
    cleaned_string = re.sub(pattern, ' ', input_string)
    return cleaned_string

def split_camel_case(input_string):
    # Split camel case by adding a space before any uppercase letter that is followed by a lowercase letter
    split_string = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', input_string)
    return split_string

def clean_column_name(col_name):
    # Strip leading/trailing spaces, convert to lowercase, split camel case, and remove invalid characters
    col_name = col_name.strip()
    col_name = split_camel_case(col_name)
    col_name = col_name.lower()
    col_name = remove_invalid_characters(col_name)
    # Reduce multiple spaces to a single space
    col_name = re.sub(r'\s+', ' ', col_name)
    return col_name

def clean_df(df):

    def clean_element(x):
        if is_null_value(x):
            return None
        if isinstance(x, str):
            val = remove_invalid_characters(x.strip().lower())
            val = split_camel_case(val)
            if val != "":
                return val
            else:
                return None
        return x

    df = df.apply(lambda col: col.apply(clean_element))

    return df


def detect_column_type(col, key_threshold=0.8, numeric_threshold=0.95, gdc_numerical=True):

    if "gene" in col.name.lower():
        # TODO, implement a less naive approach
        return "gene"
    
    if "date" in col.name.lower():
        # TODO, implement a less naive approach
        return "date"

    unique_values = col.dropna().unique()

    if len(unique_values)/len(col) > key_threshold and col.dtype not in [np.float64, np.float32, np.float16 ]:
        # columns with many distinct values are considered as "keys"
        return "key"
    
    if gdc_numerical and (col.dtype in [np.float64, np.int64] or  len(unique_values) == 0 ):
        return "numeric"

    numeric_unique_values = pd.Series(
        pd.to_numeric(unique_values, errors='coerce'))
    numeric_unique_values = numeric_unique_values.dropna()

    if not numeric_unique_values.empty:
        if len(numeric_unique_values) / len(unique_values) > numeric_threshold:
            if len(numeric_unique_values) > 2:
                return "numeric"
            else:
                unique_values_as_int = set(map(int, unique_values))
                if unique_values_as_int.issubset({0, 1}):
                    return "binary"
                else:
                    return "numeric"

    if len(unique_values) == 2 and all(is_binary_value(val) for val in unique_values):
        return "binary"
    else:
        return "categorical"
    
    raise ValueError(f"Could not detect type for column {col.name}")


def get_type2columns_map(df):
    # TODO: add more types, maybe semantic types
    types2columns_map = {}
    types2columns_map["key"] = []
    types2columns_map["numeric"] = []
    types2columns_map["categorical"] = []
    types2columns_map["binary"] = []
    types2columns_map["gene"] = []
    types2columns_map["date"] = []

    for col in df.columns:
        col_type = detect_column_type(df[col])
        types2columns_map[col_type].append(col)

    return types2columns_map

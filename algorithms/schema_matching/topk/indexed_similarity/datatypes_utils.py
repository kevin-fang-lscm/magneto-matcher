import pandas as pd
import numpy as np

NULL_REPRESENTATIONS = {
    "not allowed to collect",
    "not reported",
    "unknown",
    "not otherwise specified",
    "nos",
    "not applicable",
    "na",
    "not available",
    "n/a",
    "none",
    "null",
    "",
    " ",
    "missing",
    "unspecified",
    "undetermined",
    "not collected",
    "not recorded",
    "not provided",
    "no data",
    "unavailable",
    "empty",
    "undefined",
    "not defined",
    "0",
    "other, specify",
    "other",
    None,
    np.nan,
    pd.NaT,
    pd.NA,
    pd.NaT
}

BINARY_VALUES = {
    "yes", "no",
    "true", "false",
    "1", "0",
    "present", "absent",
    "positive", "negative",
    "detected", "not detected",
    "normal", "abnormal",
    "enabled", "disabled",
    "active", "inactive",
    "open", "closed",
    "success", "failure",
    "on", "off",
    "approved", "rejected",
    "included", "excluded",
    "passed", "failed",
    "accepted", "denied"
}


def is_null(value):
    if isinstance(value, str):
        value = value.lower()
    return value in NULL_REPRESENTATIONS


def is_binary(value):
    if isinstance(value, str):
        value = value.lower()
    return value in BINARY_VALUES

# def clean_null_cells(df):
#     return df.applymap(lambda x: None if is_null(x) else x)
def clean_null_cells(df):
    return df.apply(lambda col: col.map(lambda x: None if is_null(x) else x))
    


def detect_column_type(col):

    col = col[~col.apply(is_null)]

    if pd.api.types.is_numeric_dtype(col) and set(col.unique()).issubset({0, 1}):
        return 'binary'

    unique_values = set([str(val).lower() for val in col.unique()])

    if len(unique_values) == 2 and all(is_binary(val) for val in unique_values):
        return 'binary'

    if pd.api.types.is_numeric_dtype(col):
        return 'numeric'

    return 'categorical'


# df = pd.DataFrame({
#     'numeric_col': [1, 2, 3, 4],
#     'binary_col': ['yes', 'no', 'yes', 'no'],
#     'categorical_col': ['a', 'b', 'c', 'a'],
#     'binary_numeric_col': [1, 0, 1, 0],
#     'null_col': ['Not Reported', 'unknown', 'yes', 'nO']
# })

# print(detect_column_type(df['numeric_col']))
# print(detect_column_type(df['binary_col']))
# print(detect_column_type(df['categorical_col']))
# print(detect_column_type(df['binary_numeric_col']))
# print(detect_column_type(df['null_col']))

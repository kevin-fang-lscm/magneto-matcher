import pandas as pd
import os
import sys
from valentine import valentine_match


project_path = os.getcwd()
sys.path.append(os.path.join(project_path))

from algorithms.schema_matching.magneto import Magneto


def get_default_dataframes():
    df1 = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})

    df2 = pd.DataFrame({"id": [4, 5, 6], "name": ["David", "Eve", "Frank"]})

    return df1, df2


def run_matchmaker(file1=None, file2=None):
    if file1 and file2:
        try:
            df1 = pd.read_csv(file1)
            df2 = pd.read_csv(file2)
            print(f"Loaded data from {file1} and {file2}.")
        except Exception as e:
            print(f"Error loading CSV files: {e}")
            print("No files provided. Using default dataframes.")
            df1, df2 = get_default_dataframes()
    else:
        print("No files provided. Using default dataframes.")
        df1, df2 = get_default_dataframes()

    matcher = Magneto()

    matches = valentine_match(df1, df2, matcher)

    print("Matches:")
    for m in matches:
        print(m, matches[m])


if __name__ == "__main__":
    csv1 = sys.argv[1] if len(sys.argv) > 1 else None
    csv2 = sys.argv[2] if len(sys.argv) > 2 else None
    run_matchmaker(csv1, csv2)

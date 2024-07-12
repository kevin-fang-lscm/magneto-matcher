import pandas as pd
import numpy as np
import random
from abc import ABC, abstractmethod


DEFAULT_CONTEXT_COLUMNS = 0
DEFAULT_CONTEXT_ROWS = 0

MAXIMUM_STRING_LENGTH = 50

DEFAULT_VALUE_SAMPLE_SIZE = 5


def trim_string(s, max_length=MAXIMUM_STRING_LENGTH):
    s = str(s)
    if len(s) > max_length:
        return s[:max_length] + '...'
    return s


class ColumnSummary(ABC):
    # Generates a text summary of a table column
    @abstractmethod
    def summarize(self, df, col_name, col_values, table_name=None):
        pass


class NameAndValueSampleColumnSummary(ColumnSummary):
    # Default implementation of ColumnSummary
    def summarize(self, df, col_name, col_values, table_name=None, include_col_type=True):

        if include_col_type:
            col_type = pd.api.types.infer_dtype(col_values)
            col_type = f"({col_type})" if col_type else ""
        else:
            col_type = ""

        col_values = np.array(col_values)
        summary = f"{col_name} {col_type} attribute of {table_name} table, " if table_name else f"{col_name} {col_type} attribute, "

        unique_vals = pd.Series(col_values).nunique()
        total_vals = len(col_values)
        if (unique_vals <= DEFAULT_VALUE_SAMPLE_SIZE):
            chosen_val = list(set(col_values))
            chosen_val = [trim_string(val) for val in chosen_val]
            summary += f"with unique values: {', '.join(chosen_val)}"
        else:
            random_sample = random.sample(list(col_values), min(
                DEFAULT_VALUE_SAMPLE_SIZE, total_vals))
            chosen_val = random_sample
            chosen_val = [trim_string(val) for val in chosen_val]
            summary += (f"with values such as: {', '.join(chosen_val)} "
                        f"(Unique values count: {unique_vals})")

        return summary


class ComplexColumnSummary(ColumnSummary):
    # Default implementation of ColumnSummary
    def summarize(self, df, col_name, col_values, table_name=None, include_col_type=True):

        column = df[col_name]
        dtype = column.dtype

        col_type = pd.api.types.infer_dtype(col_values)
        col_values = np.array(col_values)

        # Basic info
        text = f"Column Name: {col_name}\n"
        text += f"Data Type: {col_type}\n"

        # Basic statistics
        text += f"Number of Rows: {len(col_values)}\n"
        text += f"Number of Unique Values: {column.nunique()}\n"
        text += f"Percentage of Missing Values: {(column.isnull().sum() / len(column)) * 100:.2f}%\n"

        unique_vals = pd.Series(col_values).nunique()
        total_vals = len(col_values)

        if (unique_vals <= DEFAULT_VALUE_SAMPLE_SIZE):
            chosen_val = list(set(col_values))
            chosen_val = [trim_string(val) for val in chosen_val]
        else:
            random_sample = random.sample(list(col_values), min(
                DEFAULT_VALUE_SAMPLE_SIZE, total_vals))
            chosen_val = random_sample
            chosen_val = [trim_string(val) for val in chosen_val]

        # Sample values
        text += f"Sample Values: {', '.join(map(str,chosen_val))}\n"

        # Unique values (for object and category types)
        if col_type == 'object' or col_type == 'string':
            unique_values = column.value_counts().head(unique_vals)
            text += f"Top {unique_vals} Unique Values (Value: Count): "
            text += ', '.join([f"'{k}': {v}" for k,
                               v in unique_values.items()]) + "\n"

        # Numerical statistics (for numeric types)
        if pd.api.types.is_numeric_dtype(dtype):
            # print(col_values)
            text += f"Min: {column.min()}, Max: {column.max()}, Mean: {column.mean(): .2f}, Median: {column.median()}\n"
            text += f"Standard Deviation: {column.std():.2f}\n"

            # Add quantile information
            # quantiles = col_values.quantile([0.25, 0.75]).tolist()
            # text += f"1st Quartile: {quantiles[0]:.2f}, 3rd Quartile: {quantiles[1]:.2f}\n"

        # Date statistics (for datetime types)
        if pd.api.types.is_datetime64_any_dtype(dtype):
            text += f"Earliest Date: {column.min()}, Latest Date: {column.max()}\n"
            text += f"Date Range: {(column.max() - column.min()).days} days\n"

        return text.strip()


class Column2TextOrganizer(ABC):
    # Abstract base class for column to text conversion
    # Implement this class in ase you want to customize the way a column is converted to text
    # For example, if you want to use different order of values from the table summary
    @abstractmethod
    def transform(self, df, col_name, col_values, context_rows, context_columns, col_summary_impl, table_name=None):
        pass


class DefaultColumn2TextOrganizer(Column2TextOrganizer):
    def transform(self, df, col_name, col_values, context_rows, context_columns, col_summary_impl, table_name=None):
        col_text = col_summary_impl.summarize(
            df, col_name, col_values, table_name)
        if context_rows and context_columns:
            row_texts = []
            for i in range(len(context_rows)):
                row_values = [f"{context_columns[j]}: {trim_string(context_rows[i][j])}" for j in range(len(context_columns))]
                # Include the value of the column in context
                row_values.append(
                    f"{col_name}: {trim_string(col_values[i])}")
                row_texts.append(" | ".join(row_values))
            col_text += ". Table samples: " + "; ".join(row_texts)
        return col_text


class ColumnContextSelector(ABC):
    # Decides which columns to use as context for a target column
    @abstractmethod
    def choose(self, df, target_column, num_context_columns):
        pass


class SamplingColumnContextSelector(ColumnContextSelector):
    # choose a random set of columns as context
    def choose(self, df, target_column, num_context_columns):
        columns = df.columns
        other_columns = [c for c in columns if c != target_column]
        return random.sample(other_columns, min(num_context_columns, len(other_columns)))


class RowContextSelector(ABC):
    @abstractmethod
    def choose(self, dff, chosen_context_columns, num_context_rows):
        pass


class SamplingRowContextSelector(RowContextSelector):
    def choose(self, dff, chosen_context_columns, num_context_rows):
        return [dff[chosen_context_columns].iloc[i].tolist() for i in random.sample(range(len(dff)), min(num_context_rows, len(dff)))]


class Dataset2Text():
    # Main Dataframe-to-Text conversion class
    def __init__(self, num_context_columns=DEFAULT_CONTEXT_COLUMNS, num_context_rows=DEFAULT_CONTEXT_ROWS,
                 column2text_impl=DefaultColumn2TextOrganizer(),
                 choose_context_columns_impl=SamplingColumnContextSelector(),
                 choose_context_row_values_impl=SamplingRowContextSelector(),
                 col_summary_impl=NameAndValueSampleColumnSummary(),
                 table_name=None):

        self.num_context_columns = num_context_columns
        self.num_context_rows = num_context_rows
        self.column2text_impl = column2text_impl
        self.choose_context_columns_impl = choose_context_columns_impl
        self.choose_context_row_values_impl = choose_context_row_values_impl
        self.col_summary_impl = col_summary_impl
        self.table_name = table_name
        self.texts = []
        self.col2Text = {}

    def _reset(self):
        self.texts = []
        self.col2Text = {}

    def transform(self, dff):
        self._reset()

        for col in dff.columns:
            chosen_context_columns = self.choose_context_columns_impl.choose(
                dff, col, self.num_context_columns)
            context_rows_values = self.choose_context_row_values_impl.choose(
                dff, chosen_context_columns, self.num_context_rows)
            transformed = self.column2text_impl.transform(dff, col, col_values=dff[col].values, context_rows=context_rows_values,
                                                          context_columns=chosen_context_columns, col_summary_impl=self.col_summary_impl, table_name=self.table_name)

            self.texts.append(transformed)
            self.col2Text[col] = transformed
        return self.texts, self.col2Text

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


# Example usage
if __name__ == "__main__":
    # Create a sample dataframe for testing
    data = {
        'Column1': ['A', 'B', 'A', 'C', 'A'],
        'Column2': [1, 2, 3, 4, 5],
        'Column3': ['X', 'Y', 'X', 'Z', 'X'],
        'Column4': [10.1, 11.2, 12.3, 13.4, 14.5]
    }

    dataset = Dataset2Text(num_context_columns=0, num_context_rows=0,
                           col_summary_impl=ComplexColumnSummary())

    df = pd.DataFrame(data)
    dataset.transform(df)

    for sentence in dataset:
        print(sentence)

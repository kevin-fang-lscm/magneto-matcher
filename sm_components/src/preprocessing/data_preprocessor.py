import pandas as pd


class DataPreprocessor:
    def __init__(self, df):
        self.df = df

    def clean_data(self):
        """
        This method cleans the DataFrame by:
        - Dropping rows with NaN values
        - (Optional) Normalize, strip white spaces, or remove duplicates, etc.
        """
        # Example cleaning process, you can expand this according to your needs.
        # self.df.dropna(inplace=True)
        self.df.columns = (
            self.df.columns.str.strip()
        )  # Strip white spaces from column names
        # Add more cleaning methods as needed, like normalization, handling missing values, etc.
        return self  # Return self for method chaining

    def normalize_columns(self):
        """
        An example of additional preprocessing like normalizing column names
        or converting units, or encoding categorical variables.
        """
        self.df.columns = (
            self.df.columns.str.lower()
        )  # Normalize column names to lowercase
        return self  # Return self for method chaining

    def get_df(self):
        """Helper method to get the processed DataFrame"""
        return self.df

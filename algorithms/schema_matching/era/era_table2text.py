import pandas as pd
from abc import ABC, abstractmethod
import pandas as pd

class ColumnToStringTransformer(ABC):
    @abstractmethod
    def transform(self, column: pd.Series) -> str:
        pass

class DefaultColumnToStringTransformer(ColumnToStringTransformer):
    def transform(self, column: pd.Series) -> str:
        return f"{column.name}: {column.dtype}\n" + \
               f"Sample values: {', '.join(map(str, column.head(3)))}\n" + \
               f"Unique values: {column.nunique()}"
    



def column_to_text(df: pd.DataFrame, column_name: str, num_samples: int = 5, num_unique: int = 10) -> str:
    column = df[column_name]
    dtype = column.dtype
    
    # Basic info
    text = f"Column Name: {column_name}\n"
    text += f"Data Type: {dtype}\n"
    
    # Basic statistics
    text += f"Number of Rows: {len(column)}\n"
    text += f"Number of Unique Values: {column.nunique()}\n"
    text += f"Percentage of Missing Values: {(column.isnull().sum() / len(column)) * 100:.2f}%\n"
    
    # Sample values
    text += f"Sample Values: {', '.join(map(str, column.sample(num_samples, replace=True)))}\n"
    
    # Unique values (for object and category types)
    if dtype == 'object' or dtype == 'category':
        unique_values = column.value_counts().head(num_unique)
        text += f"Top {num_unique} Unique Values (Value: Count): "
        text += ', '.join([f"'{k}': {v}" for k, v in unique_values.items()]) + "\n"
    
    # Numerical statistics (for numeric types)
    if pd.api.types.is_numeric_dtype(dtype):
        text += f"Min: {column.min()}, Max: {column.max()}, Mean: {column.mean():.2f}, Median: {column.median()}\n"
        text += f"Standard Deviation: {column.std():.2f}\n"
        
        # Add quantile information
        quantiles = column.quantile([0.25, 0.75]).tolist()
        text += f"1st Quartile: {quantiles[0]:.2f}, 3rd Quartile: {quantiles[1]:.2f}\n"
    
    # Date statistics (for datetime types)
    if pd.api.types.is_datetime64_any_dtype(dtype):
        text += f"Earliest Date: {column.min()}, Latest Date: {column.max()}\n"
        text += f"Date Range: {(column.max() - column.min()).days} days\n"
    
    return text.strip()

def dataframe_to_text(df: pd.DataFrame) -> List[str]:
    return [column_to_text(df, column) for column in df.columns]




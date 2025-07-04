# Source: https://github.com/delftdata/valentine/blob/master/valentine/data_sources/dataframe/dataframe_table.py

from typing import List
import pandas as pd

from .base_table import BaseTable
from .dataframe_column import DataframeColumn


class DataframeTable(BaseTable):

    def __init__(self, df: pd.DataFrame, name: str):
        self.__table_name = name
        self.__columns = dict()
        self.__df = df

    @property
    def unique_identifier(self) -> str:
        return self.__table_name

    @property
    def name(self) -> str:
        return self.__table_name

    def get_columns(self) -> List[DataframeColumn]:
        if not self.__columns:
            self.__get_columns_from_df()
        return list(self.__columns.values())

    def get_df(self) -> pd.DataFrame:
        return self.__df

    @property
    def is_empty(self) -> bool:
        return self.__df.empty

    def __get_columns_from_df(self):
        for column_name, column_data in self.__df.items():
            data = list(column_data.dropna().values)
            d_type = self.get_data_type(data, str(column_data.dtype))
            self.__columns[column_name] = DataframeColumn(
                column_name, data, d_type, self.unique_identifier
            )

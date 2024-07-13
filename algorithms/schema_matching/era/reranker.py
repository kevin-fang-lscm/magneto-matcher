from typing import Dict, List, Tuple
from valentine.algorithms.base_matcher import BaseMatcher
from valentine.data_sources.base_table import BaseTable
from valentine.algorithms.match import Match
from abc import ABC, abstractmethod




class MatchReranker(ABC):
    @abstractmethod
    def rerank_matches(self, source_input: BaseTable, target_input: BaseTable, matches: Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:
        pass


class PKMatchReranker(MatchReranker):
    def _find_approximate_keys(df):
        approximate_keys = []
        for column in df.columns:
            distinct_values = df[column].dropna().unique()
            if len(distinct_values) > 0.7 * len(df) and not df[column].dtype == 'float64':
                approximate_keys.append(column)
        return approximate_keys
    # tries to find key columns in the source and target table, if they are found, they are added as a match
    def rerank_matches(self, source_input: BaseTable, target_input: BaseTable, matches: Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:
        source_df = source_input.get_df()
        target_df = target_input.get_df()

        approximate_keys = self._find_approximate_keys(source_df)
        print("Approximate keys in source table: ", approximate_keys)
        


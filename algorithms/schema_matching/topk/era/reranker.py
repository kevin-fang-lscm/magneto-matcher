import pandas as pd
from typing import Dict, List, Tuple
from valentine.algorithms.match import Match
from abc import ABC, abstractmethod




class MatchReranker(ABC):
    @abstractmethod
    def rerank_matches(self, source_input: pd.DataFrame, target_input: pd.DataFrame, matches: Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:
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
    def rerank_matches(self, source_input: pd.DataFrame, target_input: pd.DataFrame, matches: Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:
        

        approximate_keys = self._find_approximate_keys(source_input)
        print("Approximate keys in source table: ", approximate_keys)


class ZeroShotColumnClassificationReranker(MatchReranker):

    def rerank_matches(self, source_input: pd.DataFrame, target_input: pd.DataFrame, matches: Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:
        

        cols_in_matches = sorted(list(set([match[0][1] for match in matches.keys()])))

        for col in cols_in_matches:
            print("Column: ", col)
            matches_for_col = {k[1][1]: v for k, v in matches.items() if k[0][1] == col}
            print("\t Matches for column: ", matches_for_col)



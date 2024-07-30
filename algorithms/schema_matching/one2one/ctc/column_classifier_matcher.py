from typing import Dict, Tuple

from valentine.algorithms.base_matcher import BaseMatcher
from valentine.algorithms.match import Match
from valentine.data_sources.base_table import BaseTable





class ColumnClassifierMatcher(BaseMatcher):

    def get_matches(self,
                    source_input: BaseTable,
                    target_input: BaseTable) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:

        source_df = source_input.get_df()
        target_df = target_input.get_df()

        matches = {}

        # Remove the pairs with zero similarity
        matches = {k: v for k, v in matches.items() if v > 0.0}

        return matches

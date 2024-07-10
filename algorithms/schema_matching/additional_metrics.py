from valentine.metrics.base_metric import Metric
from valentine.metrics.metric_helpers import *
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

@dataclass(eq=True, frozen=True)
class RecallAtTopK(Metric):
    """Recall at top K metric.

    Parameters
    ----------
    k : int
        The top K to consider for recall.
    """

    k: int

    def _filtered_matches(self, matches: List[Tuple[Tuple[str, str], Tuple[str, str]]]) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
        """Filters the matches to keep only the top K per column."""

        matches_per_col = {}
        for match in matches:
            source_col = match[0][1]
            if source_col not in matches_per_col:
                matches_per_col[source_col] = []
            matches_per_col[source_col].append(match)

        filtered_matches = []
        for col in matches_per_col:
            matches_for_col = matches_per_col[col]
            matches_per_col_sorted = sorted(matches_for_col, key=lambda x: matches[x], reverse=True)
            filtered_matches += matches_per_col_sorted[:self.k]

        return filtered_matches

    def apply(self, matches: List[Tuple[Tuple[str, str], Tuple[str, str]]], ground_truth: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Applies the recall at top K metric to a `MatcherResults` instance, given ground truth."""

        matches_set = set()
        filtered_matches = self._filtered_matches(matches)
        for fm in filtered_matches:
            fmatch = (fm[0][1], fm[1][1])
            matches_set.add(fmatch)

        ground_truth_set = set(ground_truth)

        tp = len(ground_truth_set.intersection(matches_set))
        recall = round((tp / len(ground_truth_set)), 3)

        return recall

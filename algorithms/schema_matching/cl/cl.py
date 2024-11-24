from itertools import product
from multiprocessing import get_context
from typing import Dict, Tuple

from jellyfish import (
    levenshtein_distance,
    damerau_levenshtein_distance,
    jaro_similarity,
    jaro_winkler_similarity,
    hamming_distance,
)

from valentine.algorithms.jaccard_distance import StringDistanceFunction

from valentine.algorithms.base_matcher import BaseMatcher
from valentine.algorithms.match import Match
from valentine.data_sources.base_table import BaseTable
from valentine.utils.utils import normalize_distance

from .cl_api import (
    ContrastiveLearningAPI,
)

from algorithms.download import get_cached_model_or_download


class CLMatcher(BaseMatcher):

    # "bdi-cl-v0.2"
    # "cl-reducer-v0.1"
    def __init__(self, model_name: str = "bdi-cl-v0.2", top_k=10):
        model_path = get_cached_model_or_download(model_name)
        self.top_k = top_k
        self.api = ContrastiveLearningAPI(model_path=model_path, top_k=top_k)
        self.minimum_similarity = 0.4

    def get_matches(
        self, source_input: BaseTable, target_input: BaseTable
    ) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:

        dataset = source_input.get_df()

        global_table = target_input.get_df()

        union_scopes, scopes_json = self.api.get_recommendations(dataset, global_table)
        matches = {}

        for column, scope in zip(dataset.columns, scopes_json):
            for i, topk in enumerate(scope["Top k columns"]):
                topk = scope["Top k columns"][i][0]
                score = scope["Top k columns"][i][1]
                score = float(score)
                # print(f"Matched {column} with {topk} with score {score}")
                match = Match(
                    target_input.name, topk, source_input.name, column, score
                ).to_dict
                matches.update(match)

        # Remove the pairs with zero similarity
        matches = {k: v for k, v in matches.items() if v >= self.minimum_similarity}

        return matches

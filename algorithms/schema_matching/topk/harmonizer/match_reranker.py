from sentence_transformers import SentenceTransformer
import numpy as np
from .harmonizer import Harmonizer
from valentine.algorithms.base_matcher import BaseMatcher
from valentine.data_sources.base_table import BaseTable
from valentine.algorithms.match import Match
import pandas as pd
from typing import Dict, Tuple
from .utils import get_samples
from scipy.optimize import linear_sum_assignment

import openai
from dotenv import load_dotenv
import os


class MatchReranker(Harmonizer):
    def __init__(self, columnheader_model_name=None, value_model_name=None,
                 topk=10, use_instances=False, llm_model='gpt-4o-mini'):
        super().__init__(columnheader_model_name, value_model_name, topk, use_instances)

        self.llm_model = llm_model

    def bipartite_filtering(self, initial_matches, source_table, target_table):

        source_cols = set()
        target_cols = set()
        for (col_source, col_target), score in initial_matches.items():
            col_source = col_source[1]
            col_target = col_target[1]
            source_cols.add(col_source)
            target_cols.add(col_target)

        # Map columns to indices
        source_col_to_num = {col: idx for idx, col in enumerate(source_cols)}
        target_col_to_num = {col: idx for idx, col in enumerate(target_cols)}

        # Initialize the score matrix with zeros
        score_matrix = np.zeros((len(source_cols), len(target_cols)))

        # Populate the matrix with scores from initial matches
        for (col_source, col_target), score in initial_matches.items():
            col_source = col_source[1]
            col_target = col_target[1]
            source_idx = source_col_to_num[col_source]
            target_idx = target_col_to_num[col_target]
            score_matrix[source_idx, target_idx] = score

        # print("Score Matrix:\n", score_matrix)

        row_ind, col_ind = linear_sum_assignment(score_matrix, maximize=True)
        assignment = list(zip(row_ind, col_ind))

        # print("Assignment:", assignment)

        filtered_matches = {}

        source_idx_to_col = {idx: col for col,
                             idx in source_col_to_num.items()}
        target_idx_to_col = {idx: col for col,
                             idx in target_col_to_num.items()}

        for source_idx, target_idx in assignment:
            source_col = source_idx_to_col[source_idx]
            target_col = target_idx_to_col[target_idx]
            filtered_matches[((source_table.name, source_col), (target_table.name, target_col))] = score_matrix[source_idx, target_idx]
 

        # print("Filtered Matches:", filtered_matches)
        return filtered_matches

    def get_matches(self, source_table: BaseTable, target_table: BaseTable) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:
        # Get initial matches using parent class
        initial_matches = super().get_matches(source_table, target_table)


        filtered_matches = self.bipartite_filtering(
            initial_matches, source_table, target_table)

        # Step 1: Remove all filtered_matches entries from initial_matches
        for key in filtered_matches.keys():
            initial_matches.pop(key, None)

        # Step 2: Find the minimum score in filtered_matches
        min_filtered_score = min(filtered_matches.values())

        # Step 3: Calculate the scaling factor to keep scores in initial_matches just below min_filtered_score
        initial_max_score = max(initial_matches.values())
        scaling_factor = (min_filtered_score - 0.01) / initial_max_score if initial_max_score > 0 else 1

        # Adjust scores in initial_matches, maintaining relative differences
        adjusted_initial_matches = {
            key: score * scaling_factor for key, score in initial_matches.items()
        }

        # Step 4: Combine filtered_matches and adjusted initial_matches
        # filtered_matches entries will remain at the beginning
        filtered_matches.update(adjusted_initial_matches)





        return filtered_matches

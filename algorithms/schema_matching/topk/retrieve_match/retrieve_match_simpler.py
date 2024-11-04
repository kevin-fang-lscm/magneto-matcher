import argparse
import pandas as pd
import os
import time
import json

from .retriever import ColumnRetriever
from .matcher import ColumnMatcher
from .utils import get_dataset_paths, process_tables, get_samples, default_converter, common_prefix
from .evaluation import evaluate_matches, convert_to_valentine_format

from valentine.algorithms.base_matcher import BaseMatcher
from valentine.data_sources.base_table import BaseTable
from typing import Dict, Tuple
from scipy.optimize import linear_sum_assignment
import numpy as np
from fuzzywuzzy import fuzz
import string

THRESHOLD = 0.95


class RetrieveMatchSimpler(BaseMatcher):
    def __init__(self, model_type, dataset, serialization, llm_model, filter=False, include_basic_matches=False):
        self.retriever = ColumnRetriever(
            model_type=model_type, dataset=dataset, serialization=serialization
        )
        # self.matcher = ColumnMatcher(llm_model=llm_model)
        self.filter = filter
        self.include_basic_matches = include_basic_matches

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
            filtered_matches[((source_table.name, source_col), (target_table.name,
                              target_col))] = score_matrix[source_idx, target_idx]

        # print("Filtered Matches:", filtered_matches)
        return filtered_matches

    def stability_score(self, initial_matches, filtered_matches):

        filtered_matches_sorted = {key: score for key, score in sorted(
            filtered_matches.items(), key=lambda item: item[1], reverse=True)}
        filtered_matches_list = list(filtered_matches_sorted.keys())
        initial_scores_sorted = {key: score for key, score in sorted(
            initial_matches.items(), key=lambda item: item[1], reverse=True)}
        initial_scores_list = list(initial_scores_sorted.keys())

        # for match in initial_scores_list:
        #     print(match, initial_scores_sorted[match])
        # print("\n")
        # for match in filtered_matches_list:
        #     print(match, filtered_matches_sorted[match])
        # print("\n")

        stability_score = 0
        for idx, key in enumerate(filtered_matches_list):
            if key in initial_scores_list:
                initial_rank = initial_scores_list.index(key)
                filtered_rank = idx
                rank_intersection = 1 - \
                    abs(initial_rank - filtered_rank) / \
                    len(initial_scores_list)
                stability_score += rank_intersection

        stability_score = stability_score / len(filtered_matches_list)

        return stability_score

    def update_for_basic_matches(self, matched_columns, source_table, target_table):
        source_cols = set(source_table.get_df().columns)
        target_cols = set(target_table.get_df().columns)

        def normalize_text(text):
            # Convert to lowercase and remove punctuation
            translator = str.maketrans('', '', string.punctuation)
            return text.lower().translate(translator).strip()

        #  Normalize column names
        normalized_source_cols = {normalize_text(
            col): col for col in source_cols}
        normalized_target_cols = {normalize_text(
            col): col for col in target_cols}

        for source_col in normalized_source_cols.keys():
            for target_col in normalized_target_cols.keys():
                #  Check for exact match (ignoring case and punctuation)
                if source_col == target_col:
                    original_source_col = normalized_source_cols[source_col]
                    original_target_col = normalized_target_cols[target_col]

                    # Add or update the exact match in matched_columns for source_col
                    if original_source_col in matched_columns:
                        # Filter out the target_col if it already exists with a different score
                        matched_columns[original_source_col] = [
                            (tgt, sim) for tgt, sim in matched_columns[original_source_col]
                            if tgt != original_target_col
                        ]
                        # Add the exact match at the top of the list
                        matched_columns[original_source_col].insert(
                            0, (original_target_col, 1.0))
                    else:
                        # If the source_col is not in matched_columns, add it with the exact match
                        matched_columns[original_source_col].insert(
                            0, (original_target_col, 1.0))

                    # Remove target_col from all other entries in matched_columns
                    for other_source, matches in matched_columns.items():
                        if other_source != original_source_col:
                            matched_columns[other_source] = [
                                (tgt, sim) for tgt, sim in matches if tgt != original_target_col
                            ]

        return matched_columns

    def match(self, source_table, target_table,  top_k, cand_k):
        orig_source_table, orig_target_table = source_table, target_table
        source_table = source_table.get_df()
        target_table = target_table.get_df()

        # if self.include_basic_matches:
        #     prefix_source = common_prefix(list(source_table.columns))
        #     prefix_target = common_prefix(list(target_table.columns))

        #     if prefix_source != "":
        #         source_table.columns = [col.replace(prefix_source, "") for col in source_table.columns]
        #     if prefix_target != "":
        #         target_table.columns = [col.replace(prefix_target, "") for col in target_table.columns]

        # print(orig_source_table.name,
        #     orig_target_table.name)

        start_time = time.time()
        source_values = {
            col: get_samples(source_table[col]) for col in source_table.columns
        }
        target_values = {
            col: get_samples(target_table[col]) for col in target_table.columns
        }
        matched_columns = self.retriever.find_matches(
            source_table, target_table, source_values, target_values, top_k
        )
        print("Matched Columns:", matched_columns)

        # if cand_k > 1:
        #     matched_columns = self.matcher.rematch(
        #         source_table,
        #         target_table,
        #         source_values,
        #         target_values,
        #         top_k,
        #         matched_columns,
        #         cand_k,
        #     )
        # print("Refined Matches:", matched_columns)

        # for col, candidates in matched_columns.items():
        #     print(col, " -> ",candidates)
        # print

        if self.include_basic_matches:
            self.update_for_basic_matches(
                matched_columns, orig_source_table, orig_target_table)

        # print("Matched Columns:", matched_columns)

        if self.filter:
            matched_columns = convert_to_valentine_format(
                matched_columns,
                orig_source_table.name,
                orig_target_table.name,
            )

            filtered_matches = self.bipartite_filtering(
                matched_columns, orig_source_table, orig_target_table)

            # Compute confidence scores for the filtered matches
            # confidence_score = np.mean(list(filtered_matches.values()))
            # print("Confidence Score:", confidence_score)
            # min, max = np.min(list(filtered_matches.values())), np.max(list(filtered_matches.values()))
            # print("Confidence Score:", confidence_score, min, max)

            # stability_score = self.stability_score(
            #     matched_columns, filtered_matches)
            # print("Stability Score:", stability_score)

            # Step 1: Remove all filtered_matches entries from initial_matches
            for key in filtered_matches.keys():
                matched_columns.pop(key, None)

            # Step 2: Find the minimum score in filtered_matches
            min_filtered_score = min(filtered_matches.values())

            # Step 3: Calculate the scaling factor to keep scores in initial_matches just below min_filtered_score
            initial_max_score = max(matched_columns.values())
            scaling_factor = (min_filtered_score - 0.01) / \
                initial_max_score if initial_max_score > 0 else 1

            # Adjust scores in initial_matches, maintaining relative differences
            adjusted_initial_matches = {
                key: score * scaling_factor for key, score in matched_columns.items()
            }

            # Step 4: Combine filtered_matches and adjusted initial_matches
            # filtered_matches entries will remain at the beginning
            filtered_matches.update(adjusted_initial_matches)

            matched_columns = filtered_matches
        else:
            matched_columns = convert_to_valentine_format(
                matched_columns,
                orig_source_table.name,
                orig_target_table.name,
            )

        runtime = time.time() - start_time

        return matched_columns, runtime, matched_columns

    def get_matches(self, source_table: BaseTable, target_table: BaseTable) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:
        converted_matches, runtime, matched_columns = self.match(
            source_table, target_table, 20, 20)
        return converted_matches

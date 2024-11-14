import os
import numpy as np
from typing import Dict, Tuple


from valentine.algorithms.base_matcher import BaseMatcher
from valentine.data_sources.base_table import BaseTable
from valentine.algorithms.match import Match

from .utils import clean_df, remove_invalid_characters, get_samples, convert_to_valentine_format
from .similarity_ranker import SimilarityRanker
from .llm_reranker import LLMReranker
from .bp_reranker import arrange_bipartite_matches

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MatchMaker(BaseMatcher):

    def __init__(self, topk=20, finetuned_model_path=None,  use_gpt=False):
        self.topk = topk
        self.finetuned_model_path = finetuned_model_path
        self.use_gpt = use_gpt

    def get_matches(self, source_table: BaseTable, target_table: BaseTable) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:

        self.df_source = clean_df(source_table.get_df())
        self.df_target = clean_df(target_table.get_df())

        if len(self.df_source.columns) == 0 or len(self.df_target.columns) == 0:
            return {}

        # Input similarity map
        self.input_sim_map = {col: {} for col in self.df_source.columns}

        schemaSimRanker = SimilarityRanker( self.topk)


        # SCHEMA-Based Matches with string similarity and alignment
        strBasicSimilarities = schemaSimRanker.get_str_similarity_candidates(
            self.df_source.columns,  self.df_target.columns)
        for (col_source, col_target), score in strBasicSimilarities.items():
            self.input_sim_map[col_source][col_target] = score

        if self.finetuned_model_path is not None:
            print("Using Fine-Tuned Model for Embeddings")

        
        else:
            #  SCHEMA-Based Matches with LM embeddings
            strEmbeddingSimilarities = schemaSimRanker.get_embedding_similarity_candidates(
                self.df_source,  self.df_target)
            for (col_source, col_target), score in strEmbeddingSimilarities.items():
                self.input_sim_map[col_source][col_target] = score

        # # ## just add the exact matches on top
        for source_col in self.df_source.columns:
            cand_source = remove_invalid_characters(source_col.strip().lower())
            for target_col in self.df_target.columns:
                cand_target = remove_invalid_characters(
                    target_col.strip().lower())
                if cand_source == cand_target:
                    self.input_sim_map[source_col][target_col] = 1.0

        # # Keep only the top-k entries for each column in input_sim_map
        for col_source in self.input_sim_map:
            sorted_matches = sorted(
                self.input_sim_map[col_source].items(), key=lambda item: item[1], reverse=True)
            top_k_matches = sorted_matches[:self.topk]
            self.input_sim_map[col_source] = dict(top_k_matches)

        # Wraps the matches into Valentine format
        matches = {}
        for col_input, matches_dict in self.input_sim_map.items():
            for col_target, score in matches_dict.items():
                match = Match(target_table.name, col_target,
                              source_table.name, col_input, score).to_dict
                matches.update(match)
                # print(match)

        if self.use_gpt:
            matches = self.llm_matcher(source_table, target_table, matches)
        else:
            matches = arrange_bipartite_matches(
                matches, source_table, target_table)

        return matches

    def llm_matcher(self, source_table, target_table, matches):

        orig_source_table, orig_target_table = source_table, target_table
        source_table = source_table.get_df()
        target_table = target_table.get_df()

        reranker = LLMReranker()

        source_values = {
            col: get_samples(source_table[col], 15, False) for col in source_table.columns
        }
        target_values = {
            col: get_samples(target_table[col], 15, False) for col in target_table.columns
        }

        matched_columns = {}
        for entry, score in matches.items():
            source_col = entry[0][1]
            target_col = entry[1][1]
            if source_col not in matched_columns:
                matched_columns[source_col] = [(target_col, score)]
            else:
                matched_columns[source_col].append((target_col, score))

        # print("Initial Matches:", matched_columns)

        matched_columns = reranker.rematch(
            source_table,
            target_table,
            source_values,
            target_values,
            self.topk,
            matched_columns,
            self.topk,
        )
        # print("Refined Matches:", matched_columns)

        converted_matches = convert_to_valentine_format(
            matched_columns,
            orig_source_table.name,
            orig_target_table.name,
        )

        return converted_matches

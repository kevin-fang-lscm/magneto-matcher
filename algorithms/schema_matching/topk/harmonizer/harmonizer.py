import os
import pandas as pd
from typing import Dict, Tuple
from valentine.algorithms.base_matcher import BaseMatcher
from valentine.data_sources.base_table import BaseTable
from valentine.algorithms.match import Match
from .utils import clean_df, remove_invalid_characters
from .schema_similarity_ranker import SchemaSimilarityRanker
from .value_similarity_ranker import ValueSimilarityRanker

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Harmonizer(BaseMatcher):

    def __init__(self,  columnheader_model_name=None, value_model_name=None, topk=10, use_instances=False,):

        self.use_instances = use_instances
        self.topk = topk
        self.columnheader_model_name = columnheader_model_name
        self.value_model_name = value_model_name

  

    def get_matches(self, source_table: BaseTable, target_table: BaseTable) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:
        
        self.df_source = clean_df(source_table.get_df())
        self.df_target = clean_df(target_table.get_df())

        # Return an empty dictionary if either dataframe has no columns
        if len(self.df_source.columns) == 0 or len(self.df_target.columns) == 0:
            return {}





        # Initialize the input similarity map
        self.input_sim_map = {col: {} for col in self.df_source.columns}


        schemaSimRanker = SchemaSimilarityRanker(model_name=self.columnheader_model_name)

        # SCHEMA-Based Matches with string similarity and alignment
        strBasicSimilarities = schemaSimRanker.get_str_similarity_candidates(self.df_source.columns,  self.df_target.columns)
        for (col_source, col_target), score in strBasicSimilarities.items():
            self.input_sim_map[col_source][col_target] = score

        # SCHEMA-Based Matches with LM embeddings
        strEmbeddingSimilarities = schemaSimRanker.get_embedding_similarity_candidates(self.df_source.columns,  self.df_target.columns)
        for (col_source, col_target), score in strEmbeddingSimilarities.items():
            self.input_sim_map[col_source][col_target] = score

        # # # # # Instance-Based Matches
        if self.use_instances:
            valueSimRanker = ValueSimilarityRanker(self.value_model_name)
            valueSimilarities = valueSimRanker.get_type_and_value_based_candidates(self.df_source, self.df_target)

            for (col_source, col_target), score in valueSimilarities.items():
                self.input_sim_map[col_source][col_target] = score


        # ## just add the exact matches on top
        for source_col in self.df_source.columns:
            cand_source = remove_invalid_characters(source_col.strip().lower())
            for target_col in  self.df_target.columns:
                cand_target = remove_invalid_characters(target_col.strip().lower())
                if cand_source == cand_target:
                    self.input_sim_map[source_col][target_col] = 1.0


        # Keep only the top-k entries for each column in input_sim_map
        for col_source in self.input_sim_map:
            sorted_matches = sorted(self.input_sim_map[col_source].items(), key=lambda item: item[1], reverse=True)
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

        return matches


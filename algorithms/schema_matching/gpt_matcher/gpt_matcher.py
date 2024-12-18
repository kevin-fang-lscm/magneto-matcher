import random
from typing import Dict, Tuple

import tiktoken
from valentine.algorithms.base_matcher import BaseMatcher
from valentine.data_sources.base_table import BaseTable

from algorithms.schema_matching.magneto.llm_reranker import LLMReranker
from algorithms.schema_matching.magneto.utils import (
    convert_to_valentine_format,
    get_samples,
)


class GPTMatcher(BaseMatcher):
    def __init__(self, llm_model="gpt-4o-mini", sample_size=10, random_order=False):
        self.llm_model = llm_model
        self.sample_size = sample_size
        self.target_columns_random_order = random_order

    def num_tokens_from_string(self, string, encoding_name="gpt-4"):
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def get_matches(
        self, source_table: BaseTable, target_table: BaseTable
    ) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:
        source_name = source_table.name
        target_name = target_table.name

        source_table = source_table.get_df()
        target_table = target_table.get_df()

        refined_matches = []

        source_columns = source_table.columns
        target_columns = target_table.columns

        reranker = LLMReranker()

        source_values = {
            col: get_samples(source_table[col], self.sample_size)
            for col in source_table.columns
        }
        target_values = {
            col: get_samples(target_table[col], self.sample_size)
            for col in target_table.columns
        }

        matched_columns = {}

        target_col_list = [(col, 1.0) for col in target_columns]

        if self.target_columns_random_order:
            random.shuffle(target_col_list)

        for source_col in source_columns:
            matched_columns[source_col] = target_col_list

        matched_columns = reranker.rematch(
            source_table, target_table, source_values, target_values, matched_columns,
        )

        converted_matches = convert_to_valentine_format(
            matched_columns, source_name, target_name,
        )

        return converted_matches

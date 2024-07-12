import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
from valentine.algorithms.base_matcher import BaseMatcher
from valentine.algorithms.match import Match
from valentine.data_sources.base_table import BaseTable
from algorithms.schema_matching.era.table2text import Dataset2Text
from valentine import valentine_match


class EmbedRetrieveAlign(BaseMatcher):
    def __init__(self, model_name: str, column_transformer: Dataset2Text = Dataset2Text(num_context_columns=0, num_context_rows=0),  top_k: int = 1):
        self.column_transformer = column_transformer
        self.top_k = top_k
        print(f"Loading model '{model_name}'")
        try:
            # Try to load the model from sentence-transformers
            self.model = SentenceTransformer(model_name)
        except ValueError:
                raise ValueError(f"Model '{model_name}' not found in sentence-transformers or on disk.")

    def _get_column_embeddings(self, table: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
        columns = table.columns
        column_strings, col2Text = self.column_transformer.transform(table)
        embeddings = self.model.encode(column_strings)
        return columns, embeddings

    def get_matches(self, source_input: BaseTable, target_input: BaseTable) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:

        source_df = source_input.get_df()
        target_df = target_input.get_df()

        source_columns, source_embeddings = self._get_column_embeddings(
            source_df)

        target_columns, target_embeddings = self._get_column_embeddings(
            target_df)

        similarity_matrix = cosine_similarity(
            source_embeddings, target_embeddings)

        matches = {}
        for i, source_column in enumerate(source_columns):
            top_k_indices = np.argsort(
                similarity_matrix[i])[- self.top_k:][::-1]
            top_k_scores = similarity_matrix[i][top_k_indices]

            for idx, score in zip(top_k_indices, top_k_scores):
                score = float(score)

                match = Match(target_input.name, target_columns[idx],
                              source_input.name, source_column,
                              score).to_dict
                matches.update(match)

        matches = {k: v for k, v in matches.items() if v > 0.0}

        return matches

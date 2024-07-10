import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
from valentine.algorithms.base_matcher import BaseMatcher
from valentine.algorithms.match import Match
from valentine.data_sources.base_table import BaseTable
from algorithms.schema_matching.era.era_table2text import ColumnToStringTransformer, DefaultColumnToStringTransformer
from valentine import valentine_match

class EmbedRetrieveAlign(BaseMatcher):
    def __init__(self, column_transformer: ColumnToStringTransformer = DefaultColumnToStringTransformer(), model_name: str = 'all-mpnet-base-v2', top_k: int = 1):
        self.column_transformer = column_transformer
        self.top_k = top_k

        try:
            # Try to load the model from sentence-transformers
            self.model = SentenceTransformer(model_name)
        except ValueError:
            # If the model is not found in sentence-transformers, try to load it from disk
            if os.path.exists(model_name):
                self.model = SentenceTransformer(model_name)
            else:
                raise ValueError(f"Model '{model_name}' not found in sentence-transformers or on disk.")


    def _get_column_embeddings(self, table: BaseTable) -> Tuple[List[str], np.ndarray]:
        columns = table.columns
        column_strings = [self.column_transformer.transform(table[col]) for col in columns]
        embeddings = self.model.encode(column_strings)
        return columns, embeddings

    def get_matches(self, source_input: BaseTable, target_input: BaseTable) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:
        
        source_df = source_input.get_df()
        target_df = target_input.get_df()
        
        source_columns, source_embeddings = self._get_column_embeddings(source_df)
        target_columns, target_embeddings = self._get_column_embeddings(target_df)

        similarity_matrix = cosine_similarity(source_embeddings, target_embeddings)

        matches = {}
        for i, source_column in enumerate(source_columns):
            top_k_indices = np.argsort(similarity_matrix[i])[- self.top_k:][::-1]
            top_k_scores = similarity_matrix[i][top_k_indices]

            for idx, score in zip(top_k_indices, top_k_scores):
                score = float(score)

                match = Match(target_input.name, target_columns[idx],
                        source_input.name, source_column,
                        score).to_dict
                matches.update(match)

        matches = {k: v for k, v in matches.items() if v > 0.0}

        return matches



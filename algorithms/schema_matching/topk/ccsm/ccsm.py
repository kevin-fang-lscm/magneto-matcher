import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
from polyfuzz.models import EditDistance, TFIDF, Embeddings, RapidFuzz
from polyfuzz import PolyFuzz
from jellyfish import jaro_winkler_similarity
from flair.embeddings import TransformerWordEmbeddings, WordEmbeddings
from valentine.algorithms.base_matcher import BaseMatcher
from valentine.data_sources.base_table import BaseTable
from valentine.algorithms.match import Match
from algorithms.schema_matching.topk.ccsm.ccsm_utils import compute_term_similarity
import tqdm
from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize


def infer_type(series: pd.Series) -> str:
    """Infer the type of a pandas series."""
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    elif pd.api.types.is_timedelta64_dtype(series):
        return "timedelta"
    elif series.dtype == 'object':
        return "string"
    else:
        return "unknown"


def normalize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize numeric columns in a pandas DataFrame."""
    for col in df.columns:
        if infer_type(df[col]) == 'numeric':
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df


class Combiner(ABC):
    '''Abstract class for combining/aggregating similarities'''
    @abstractmethod
    def combine(self, similarities: List[float]) -> float:
        pass


class StatisticalCombiner(Combiner):
    def __init__(self, mode: str = 'mean'):
        self.mode = mode

    def combine(self, similarities: List[float]) -> float:
        if self.mode == 'mean':
            return sum(similarities) / len(similarities)
        elif self.mode == 'max':
            return max(similarities)
        elif self.mode == 'min':
            return min(similarities)
        elif self.mode == 'weighted':
            if len(similarities) == 2:
                weighted_similarity = (similarities[0] * 0.75) + (similarities[1] * 0.25)
                return weighted_similarity
            else:
                return sum(similarities) / len(similarities)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")


class SimilarityCalculator(ABC):
    @abstractmethod
    def compute_similarity(self, col1: str, col2: str, df1: pd.DataFrame, df2: pd.DataFrame) -> float:
        pass


class ColumnNameSimilarity(SimilarityCalculator):
    def __init__(self, embedding_model: SentenceTransformer = None):
        self.embedding_model = embedding_model

    def compute_similarity(self, col1: str, col2: str, df1: pd.DataFrame, df2: pd.DataFrame) -> float:
        similarities_dict = compute_term_similarity(col1, col2, self.embedding_model)
        max_similarity = max(similarities_dict.values())
        return max_similarity


class ColumnValueSimilarity(SimilarityCalculator):
    def __init__(self, min_similarity: float = 0.5):
        self.min_similarity = min_similarity

        self.tfidf_matcher = TFIDF(n_gram_range=(1, 3), min_similarity=self.min_similarity, model_id="TF-IDF")
        self.jellyfish_matcher = EditDistance(n_jobs=1, scorer=jaro_winkler_similarity)
        self.rapidfuzz_matcher = RapidFuzz(n_jobs=1)

        self.matchers = [self.tfidf_matcher, self.jellyfish_matcher, self.rapidfuzz_matcher]

    def compute_similarity(self, col1: str, col2: str, df1: pd.DataFrame, df2: pd.DataFrame) -> float:
        source_data = list(set(df1[col1].dropna().tolist()))
        target_data = list(set(df2[col2].dropna().tolist()))

        source_data = [str(data) for data in source_data]
        target_data = [str(data) for data in target_data]

        models = PolyFuzz(self.matchers).match(source_data, target_data)

        model_similarities_dict = {}
        for model in models.get_matches():
            model_similarities = models.get_matches(model)
            model_similarities = model_similarities[
                model_similarities['From'].notnull() & model_similarities['To'].notnull()
            ]
            mean_similarity = model_similarities['Similarity'].median()
            model_similarities_dict[model] = mean_similarity

        max_similarity = max(model_similarities_dict.values())
        return max_similarity


class CombinedColumnSimilarityMatcher(BaseMatcher):
    def __init__(self, min_similarity: float = 0.5, top_k: int = 1):
        self.min_similarity = min_similarity
        self.top_k = top_k
        self.matchers = [ColumnNameSimilarity(), ColumnValueSimilarity()]
        self.combiner = StatisticalCombiner(mode='mean')

    def get_matches(self, source_input: BaseTable, target_input: BaseTable) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:
        source_df = source_input.get_df()
        target_df = target_input.get_df()

        matches = {}
        for col1 in tqdm.tqdm(source_df.columns):
            col_matches = {}
            for col2 in target_df.columns:
                if infer_type(source_df[col1]) != infer_type(target_df[col2]):
                    continue

                similarities = [matcher.compute_similarity(col1, col2, source_df, target_df) for matcher in self.matchers]
                similarity = self.combiner.combine(similarities)
                col_matches[col2] = similarity

            if col_matches:
                sorted_matches = sorted(col_matches.items(), key=lambda item: item[1], reverse=True)
                top_k_matches = sorted_matches[:self.top_k]
                for match in top_k_matches:
                    best_match, best_score = match
                    match_dict = Match(target_input.name, best_match, source_input.name, col1, best_score).to_dict
                    matches.update(match_dict)

        matches = {k: v for k, v in matches.items() if v > 0.0}
        return matches

import os
import pandas as pd
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
import faiss

from valentine.algorithms.base_matcher import BaseMatcher

from indexes import SimilarityIndex
from datatypes_utils import detect_column_type, is_null
from utils import normalize_scores, are_similar_ignore_case_and_punctuation, are_equal_ignore_case_and_punctuation


class IndexedSimilarityMatcher(BaseMatcher):
    def __init__(self, top_k: int = 20):

        self.top_k = top_k

        self.source_categorical_df = None
        self.source_numeric_df = None
        self.source_binary_df = None

        self.target_categorical_df = None
        self.target_numeric_df = None
        self.target_binary_df = None

        self.catdomain_inverted_index = None
        self.catcol2indexes_dict = {}

    def _organize_columns_by_type(self, source_df, target_df):

        source_categorical_columns = []
        source_numeric_columns = []
        source_binary_columns = []

        for col in source_df.columns:
            if detect_column_type(source_df[col]) == 'binary':
                source_binary_columns.append(col)
            elif detect_column_type(source_df[col]) == 'numeric':
                source_numeric_columns.append(col)
            else:
                source_categorical_columns.append(col)

        self.source_categorical_df = source_df[source_categorical_columns]
        self.source_numeric_df = source_df[source_numeric_columns]
        self.source_binary_df = source_df[source_binary_columns]

        target_categorical_columns = []
        target_numeric_columns = []
        target_binary_columns = []

        for col in target_df.columns:
            if detect_column_type(target_df[col]) == 'binary':
                target_binary_columns.append(col)
            elif detect_column_type(target_df[col]) == 'numeric':
                target_numeric_columns.append(col)
            else:
                target_categorical_columns.append(col)

        self.target_categorical_df = target_df[target_categorical_columns]
        self.target_numeric_df = target_df[target_numeric_columns]
        self.target_binary_df = target_df[target_binary_columns]

    def _get_categorical_candidate_columns(self, column, clean_nulls=True):

        values = self.source_categorical_df[column].dropna().unique()
        if clean_nulls:
            values = [value for value in values if not is_null(value)]

        candidates_columns = set()

        print(f'Querying for {column} values')
        for value in values:
            results = self.catdomain_inverted_index.query(value)
            for result in results:
                candidates_columns.update(result['columns'])
        return list(candidates_columns)

    def _rerank_categorical_candidates(self, column, candidates_columns, clean_nulls=True):

        candidate_col_max_sim = {}
        for candidate_col in candidates_columns:

            if candidate_col not in self.catcol2indexes_dict:
                # only retrieve it once
                self.catcol2indexes_dict[candidate_col] = self.catdomain_inverted_index.get_col_index(
                    candidate_col)

            col_values, col_index = self.catcol2indexes_dict[candidate_col]

            max_sim = {}
            values = self.source_categorical_df[column].dropna().unique()
            if clean_nulls:
                values = [value for value in values if not is_null(value)]

            for value in values:
                if value in col_values:
                    max_sim[value] = 1.0
                else:
                    query_embedding = self.catdomain_inverted_index.encode_text(
                        value).astype('float32')
                    faiss.normalize_L2(query_embedding.reshape(1, -1))
                    topk = 1
                    D, I = col_index.search(
                        query_embedding.reshape(1, -1), topk)
                    similarity = D[0][0]
                    max_sim[value] = similarity
            sum_similarities = sum(max_sim.values())
            candidate_col_max_sim[candidate_col] = sum_similarities
            # mean_similarities = sum_similarities / len(max_sim)
            # candidate_col_max_sim[candidate_col] = mean_similarities

        sorted_candidate_col_max_sim = sorted(
            candidate_col_max_sim.items(), key=lambda x: x[1], reverse=True)
        return sorted_candidate_col_max_sim

    def get_matches(self, source_df: pd.DataFrame, target_df: pd.DataFrame, ground_truth) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:

        self._organize_columns_by_type(source_df, target_df)

        matches = {}

        self.catdomain_inverted_index = SimilarityIndex(
            self.target_categorical_df)

        for col in self.source_categorical_df:

            # if col != 'Vital.Status':
            #     continue

            source, target = [
                col_pair for col_pair in ground_truth if col_pair[0] == col][0]

            print(f'Processing column {col} with gt {source}->{target}')

            candidates_columns = self._get_categorical_candidate_columns(col)

            sorted_candidate_col_max_sim = self._rerank_categorical_candidates(
                col, candidates_columns)

            topk_entries = sorted_candidate_col_max_sim[:self.top_k]

            topk_entries = normalize_scores(topk_entries)

            # re ordering based on column name
            equal_entries = [
                entry for entry in topk_entries if are_equal_ignore_case_and_punctuation(col, entry[0])]
            equal_entries = [(entry[0], 1) for entry in equal_entries]

            similar_entries = [
                entry for entry in topk_entries if not are_similar_ignore_case_and_punctuation(col, entry[0])]
            similar_entries = [(entry[0], 1) for entry in equal_entries]

            equal_entries_out_topk = [
                (target_col, 1.0) for target_col in self.target_categorical_df.columns if are_similar_ignore_case_and_punctuation(col, target_col)]

            print('out ok topk', equal_entries_out_topk)

            non_similar_column_name = [
                entry for entry in topk_entries if entry not in equal_entries]
            non_similar_column_name = [
                entry for entry in non_similar_column_name if entry not in similar_entries]

            topk_entries = {}
            topk_entries.update(equal_entries)
            topk_entries.update(similar_entries)
            topk_entries.update(equal_entries_out_topk)
            topk_entries.update(non_similar_column_name)

            print(topk_entries)

            position = [i for i, (candidate_col, score) in enumerate(
                topk_entries.items()) if candidate_col.lower() == target.lower()]
            print(f'Position of {target} in sorted candidates: {position}')
            if len(position) == 0:
                print('Not found')

            print('\n')

        matches = {k: v for k, v in matches.items() if v > 0.0}
        return matches


def get_dataframes():
    # gdc_cat_path = './data/gdc/gdc_cat_ingt.csv'
    gdc_cat_path = './data/gdc/gdc_cat.csv'
    df_gdc = pd.read_csv(gdc_cat_path, engine='python')

    files = ['Dou.csv', 'Krug.csv', 'Clark.csv',  'Vasaikar.csv',
             'Wang.csv', 'Satpathy.csv', 'Cao.csv', 'Huang.csv', 'Gilette.csv']
    file = files[0]
    df_input = pd.read_csv(f'./data/gdc/source-tables/{file}', engine='python')

    gt_df = pd.read_csv(f'./data/gdc/ground-truth/{file}', engine='python')
    ground_truth = list(gt_df.itertuples(index=False, name=None))
    gt_gdc_cols = set([col[1] for col in ground_truth])
    gt_gdc_cols = set(gt_gdc_cols).intersection(df_gdc.columns)
    gt_gdc_cols_in_input_df = [col_pair[0]
                               for col_pair in ground_truth if col_pair[1] in gt_gdc_cols]
    df_input = df_input[gt_gdc_cols_in_input_df]

    return df_input, df_gdc, ground_truth


if __name__ == '__main__':
    df_input, df_gdc, ground_truth = get_dataframes()
    # print(df_input)
    # print(df_gdc)
    # print(ground_truth)

    matcher = IndexedSimilarityMatcher()
    matches = matcher.get_matches(df_input, df_gdc, ground_truth)

    # matcher = CombinedColumnSimilarityMatcher()
    # matches = matcher.get_matches(df1, df2)
    # print(matches)

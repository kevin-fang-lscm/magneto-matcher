import os
import pandas as pd
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
import numpy as np
import tqdm
from functools import lru_cache
import concurrent.futures
import tqdm

import torch
from transformers import AutoTokenizer, AutoModel


from valentine.algorithms.base_matcher import BaseMatcher

from preprocessing import clean_df, get_type2columns_map, clean_column_name

from embedding_utils import compute_cosine_similarity


os.environ["TOKENIZERS_PARALLELISM"] = "false"

curr_directory = os.path.dirname(os.path.realpath(__file__))
CACHE_DIR = os.path.join(curr_directory, "cache_col_names")


class IndexedSimilarityMatcher(BaseMatcher):

    def __init__(self, column_name_model='sentence-transformers/all-mpnet-base-v2'):

        self.column_name_tokenizer = AutoTokenizer.from_pretrained(
            column_name_model)
        self.column_name_model = AutoModel.from_pretrained(column_name_model)

    def _get_column_name_embeddings_batch(self, column_names, batch_size=32):
        embeddings = []
        for i in range(0, len(column_names), batch_size):
            batch_column_names = column_names[i:i+batch_size]
            inputs = self.column_name_tokenizer(batch_column_names, padding=True,
                                                truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = self.column_name_model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1))
        return torch.cat(embeddings)

    def _rank_by_embedding_similarity(self,  input_cols, target_cols):

        print("Handling numeric columns")
        embeddings_input = self._get_column_name_embeddings_batch(
            input_cols)

        embeddings_target = self._get_column_name_embeddings_batch(
            target_cols)

        topk_similarity, topk_indices = compute_cosine_similarity(
            embeddings_input, embeddings_target, 10)

        for i, col1 in enumerate(input_cols):
            print(f"Top matches for {col1}:")
            for j in range(10):
                col2 = target_cols[topk_indices[i, j]]
                print('\t', col2, ' ', topk_similarity[i, j])

    def _handle_numeric_cols(self, input_cols, target_cols):
        input_cols = [clean_column_name(col) for col in input_cols]
        target_cols = [clean_column_name(col) for col in target_cols]

        self._rank_by_embedding_similarity(
            input_cols, target_cols)

    def _handle_binary_cols(self, input_cols, target_cols):
        input_cols = [clean_column_name(col) for col in input_cols]
        target_cols = [clean_column_name(col) for col in target_cols]

        self._rank_by_embedding_similarity(
            input_cols, target_cols)

    def _handle_categorical_cols(self, input_df, target_df):
        value_matrix_input = {}

        # Define a helper function for parallel processing
        def process_column(column):
            col_values = target_df[column].dropna().unique().tolist()
            return column, self._get_column_name_embeddings_batch(col_values)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Use a ThreadPoolExecutor for I/O bound tasks or ProcessPoolExecutor for CPU bound
            futures = {executor.submit(process_column, column): column for column in target_df.columns}

            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                column, col_embeddings = future.result()
                value_matrix_input[column] = col_embeddings

        return value_matrix_input

        # print(value_matrix_input)

    def get_matches(self, source_df: pd.DataFrame, target_df: pd.DataFrame, ground_truth) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:

        input = clean_df(source_df)
        target = clean_df(target_df)

        type2cols_input = get_type2columns_map(input)
        type2cols_target = get_type2columns_map(target)

        # TODO use a loop when large numbers of data types

        # numeric_colnames_input = type2cols_input['numeric']
        # numeric_cols_target = type2cols_target['numeric']
        # if len(numeric_colnames_input) > 0 and len(numeric_cols_target) > 0:
        #     self._handle_numeric_cols(numeric_colnames_input, numeric_cols_target)

        # binary_colnames_input = type2cols_input['binary']
        # binary_cols_target = type2cols_target['binary']
        # if len(binary_colnames_input) > 0 and len(binary_cols_target) > 0:
        #     self._handle_binary_cols(binary_colnames_input, binary_cols_target)

        categorical_colnames_input = type2cols_input['categorical']
        categorical_cols_target = type2cols_target['categorical']
        if len(categorical_colnames_input) > 0 and len(categorical_cols_target) > 0:
            self._handle_categorical_cols(input[categorical_colnames_input], target[categorical_cols_target])


def get_dataframes(file):
    # gdc_cat_path = './data/gdc/gdc_cat_ingt.csv'
    # gdc_cat_path = './data/gdc/gdc_cat.csv'
    gdc_cat_path = './data/gdc/gdc_num_cat.csv'

    df_gdc = pd.read_csv(gdc_cat_path, encoding='utf-8', engine='python')

    df_input = pd.read_csv(f'./data/gdc/source-tables/{file}')

    gt_df = pd.read_csv(
        f'./data/gdc/ground-truth/{file}', encoding='utf-8', engine='python')
    ground_truth = list(gt_df.itertuples(index=False, name=None))

    gt_gdc_cols = set([col[1] for col in ground_truth])
    gt_gdc_cols = set(gt_gdc_cols).intersection(df_gdc.columns)
    gt_gdc_cols_in_input_df = [col_pair[0]
                               for col_pair in ground_truth if col_pair[1] in gt_gdc_cols]

    df_input = df_input[gt_gdc_cols_in_input_df]

    return df_input, df_gdc, ground_truth


if __name__ == '__main__':

    files = ['Dou.csv', 'Krug.csv', 'Clark.csv',  'Vasaikar.csv',
             'Wang.csv', 'Satpathy.csv', 'Cao.csv', 'Huang.csv', 'Gilette.csv']
    # file = files[-1]

    files = ['Krug.csv']

    # positions_global = []
    results_global = []
    for file in files:
        print("File: ", file)

        df_input, df_gdc, ground_truth = get_dataframes(file)

        print(df_input.shape)
        print(df_gdc.shape)

        matcher = IndexedSimilarityMatcher()
        results = matcher.get_matches(df_input, df_gdc, ground_truth)

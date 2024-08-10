import os
import pandas as pd
from typing import Dict, Tuple
import tqdm
import concurrent.futures
import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from valentine.algorithms.base_matcher import BaseMatcher
from preprocessing import clean_df, get_type2columns_map, clean_column_name
from embedding_utils import compute_cosine_similarity, compute_cosine_similarity_simple

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class IndexedSimilarityMatcher(BaseMatcher):

    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2'):
        # TODO: Add support for different models dependending on the input domain
        # TODO: Use different models for column name matching, value matching, etc.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def _get_embeddings(self, texts, batch_size=32):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, padding=True,
                                    truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1))
        return torch.cat(embeddings)

    def _column_name_matching(self, input_colnames, target_colnames):
        input_colnames = [clean_column_name(col) for col in input_colnames]
        target_colnames = [clean_column_name(col) for col in target_colnames]

        embeddings_input = self._get_embeddings(input_colnames)
        embeddings_target = self._get_embeddings(target_colnames)

        topk_similarity, topk_indices = compute_cosine_similarity_simple(
            embeddings_input, embeddings_target, 10)

        # for i, col1 in enumerate(input_colnames):
        #     print(f"Top matches for {col1}:")
        #     for j in range(10):
        #         col2 = target_colnames[topk_indices[i, j]]
        #         print('\t', col2, ' ', topk_similarity[i, j])

    def _create_column_value_embedding_dict(self, df):

        col2_val_emb = {}

        def process_column(column):
            col_values = df[column].dropna().unique().tolist()
            return column, col_values, self._get_embeddings(col_values)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(
                process_column, column): column for column in df.columns}

            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                column, col_values, col_embeddings = future.result()
                col2_val_emb[column] = (col_values, col_embeddings)
        return col2_val_emb

    def _column_value_matching(self, input_df, target_df, ground_truth, top_k=40):

        col2valemb_input = self._create_column_value_embedding_dict(input_df)
        col2valemb_target = self._create_column_value_embedding_dict(target_df)

        # flatten version of the embeddings for faster computation of cosine similarity
        all_target_embeddings = []
        all_target_values = []
        target_column_mapping = []

        # Track the start and end positions for each target column, for retrieving the embeddings from all_target_embeddings for a specific column
        current_position = 0
        target_column_boundaries = {}

        for target_col, (target_values, target_embeddings) in col2valemb_target.items():

            num_values = len(target_values)

            all_target_embeddings.append(target_embeddings)
            all_target_values.extend(target_values)
            target_column_mapping.extend([target_col] * num_values)

            target_column_boundaries[target_col] = (
                current_position, current_position + num_values)
            current_position += num_values

        # we flatten the embeddings for faster computation
        all_target_embeddings = torch.cat(all_target_embeddings)

        for input_col, (input_values, input_embeddings) in col2valemb_input.items():

            # for each column, compare to all values in target columns
            top_k_scores, top_k_indices, similarities = compute_cosine_similarity(
                input_embeddings, all_target_embeddings, top_k)
            top_k_columns = {
                target_column_mapping[idx] for row_indices in top_k_indices for idx in row_indices}

            reranked_columns = {}

            print(f"Top matches for {input_col}:")

            for target_col in list(top_k_columns):
                # TODO: save the value mappings
                (indx_begin, idx_end) = target_column_boundaries[target_col]
                similarities_col = similarities[:, indx_begin:idx_end]

                max_scores, max_positions = torch.max(similarities_col, dim=1)
                
                values_input = [input_values[idx] for idx in range(len(input_values))]
                max_values_target = [all_target_values[indx_begin + idx] for idx in max_positions]
                combinations = list(zip(values_input, max_values_target))

                print(f"Top matches for {input_col} -> {target_col}:")
                for comb, score in zip(combinations, max_scores):
                    if score > 0.5:
                        print(f"\t{comb} -> {score}")
                

                sum_max_scores = torch.sum(max_scores)

                reranked_columns[target_col] = sum_max_scores

            reranked_columns = dict(
                sorted(reranked_columns.items(), key=lambda item: item[1], reverse=True))

            # if input_col in ground_truth:
            #     match = ground_truth[input_col]
            #     position = -1
            #     if match in reranked_columns:
            #         position = list(reranked_columns.keys()).index(match)
            #         print(f"Ground truth: {
            #               input_col} -> {match} at position {position}")
            #     else:
            #         print(f"Ground truth: {input_col} -> {match} not found")

    def get_matches(self, source_df: pd.DataFrame, target_df: pd.DataFrame, ground_truth) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:

        input = clean_df(source_df)
        target = clean_df(target_df)

        type2cols_input = get_type2columns_map(input)
        type2cols_target = get_type2columns_map(target)

        # for type in ['numeric', 'binary']:
            
        #     colnames_input = type2cols_input[type]
        #     colnames_target = type2cols_target[type]

        #     if len(colnames_input) > 0 and len(colnames_target) > 0:
        #         print(f"Matching {type} columns")
        #         self._column_name_matching(colnames_input, colnames_target)

        categorical_colnames_input = type2cols_input['categorical']
        categorical_cols_target = type2cols_target['categorical']

        if len(categorical_colnames_input) > 0 and len(categorical_cols_target) > 0:
            print("Matching categorical columns")
            self._column_value_matching(
                input[categorical_colnames_input], target[categorical_cols_target],  ground_truth)

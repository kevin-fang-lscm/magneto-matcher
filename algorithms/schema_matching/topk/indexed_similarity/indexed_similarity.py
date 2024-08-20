import os
import pandas as pd
from typing import Dict, Tuple
import tqdm
import concurrent.futures
import torch
from transformers import AutoTokenizer, AutoModel
from valentine.algorithms.base_matcher import BaseMatcher
from valentine.data_sources.base_table import BaseTable
from valentine.algorithms.match import Match
from .preprocessing import clean_df, get_type2columns_map, clean_column_name
from .embedding_utils import compute_cosine_similarity, compute_cosine_similarity_simple

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class IndexedSimilarityMatcher(BaseMatcher):

    def __init__(self, config=None, use_instances=False):

        default_config = {
            'model_name': 'sentence-transformers/all-mpnet-base-v2',
            'params': {
                'column_name': {
                    'top_k': 10,
                    'minimum_similarity': 0.4,
                },
                'column_value': {
                    'top_k_sim_values': 30,
                    'minimum_similarity': 0.4,
                }
            }
        }

        self.config = config if config else default_config

        self.model_name = self.config['model_name']
        self.params = self.config['params']

        self.use_instances = use_instances

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

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
        
        input_colnames_dict = {clean_column_name(col): col for col in input_colnames}
        target_colnames_dict = {clean_column_name(col): col for col in target_colnames}
        
        
        cleaned_input_colnames = list(input_colnames_dict.keys())
        cleaned_target_colnames = list(target_colnames_dict.keys())
        
        
        embeddings_input = self._get_embeddings(cleaned_input_colnames)
        embeddings_target = self._get_embeddings(cleaned_target_colnames)
        
        
        top_k = min(self.params['column_name']['top_k'], len(cleaned_target_colnames))
        topk_similarity, topk_indices = compute_cosine_similarity_simple(
            embeddings_input, embeddings_target, top_k)
        
        
        column_similarity_map = {}
        for i, cleaned_input_col in enumerate(cleaned_input_colnames):
            original_input_col = input_colnames_dict[cleaned_input_col]
            column_similarity_map[original_input_col] = {}
            
            for j in range(top_k):
                cleaned_target_col = cleaned_target_colnames[topk_indices[i, j]]
                original_target_col = target_colnames_dict[cleaned_target_col]
                similarity = topk_similarity[i, j].item()
                
                if similarity >= self.params['column_name']['minimum_similarity']:
                    column_similarity_map[original_input_col][original_target_col] = similarity
        
        ranked_column_similarity_map = {}
        for original_input_col, matches in column_similarity_map.items():
            ranked_column_similarity_map[original_input_col] = dict(
                sorted(matches.items(), key=lambda item: item[1], reverse=True))
        
        return ranked_column_similarity_map


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

    def _column_value_matching(self, input_df, target_df):

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

        column_similarity_map = {}

        for input_col, (input_values, input_embeddings) in col2valemb_input.items():

            # for each column, compare to all values in target columns
            top_k_scores, top_k_indices, similarities = compute_cosine_similarity(
                input_embeddings, all_target_embeddings, self.config['params']['column_value']['top_k_sim_values'])
            top_k_columns = {
                target_column_mapping[idx] for row_indices in top_k_indices for idx in row_indices}

            # print(f"Top matches for {input_col}:")
            reranked_columns = {}

            for target_col in list(top_k_columns):

                # TODO: save the value mappings
                (indx_begin, idx_end) = target_column_boundaries[target_col]

                similarities_col = similarities[:, indx_begin:idx_end]

                max_scores, max_positions = torch.max(similarities_col, dim=1)

                # values_input = [input_values[idx]
                #                 for idx in range(len(input_values))]
                # max_values_target = [
                #     all_target_values[indx_begin + idx] for idx in max_positions]
                # combinations = list(zip(values_input, max_values_target))

                # for comb, score in zip(combinations, max_scores):
                #     print(f"{input_col} -> {target_col} : {comb} -> {score}")

                score_sum = torch.sum(max_scores).item()/len(max_scores)

                if score_sum >= self.config['params']['column_value']['minimum_similarity']:
                    reranked_columns[target_col] = score_sum

            reranked_columns = dict(
                sorted(reranked_columns.items(), key=lambda item: item[1], reverse=True))

            column_similarity_map[input_col] = reranked_columns

        return column_similarity_map

    def get_matches(self, source_df: BaseTable, target_df: BaseTable) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:

        input = source_df.get_df()
        target = target_df.get_df()

        input = clean_df(input)
        target = clean_df(target)

        all_matches = {}

        colnames_input = input.columns
        colnames_target = target.columns

        if len(colnames_input) > 0 and len(colnames_target) > 0:
                # print(f"Matching {type} columns")
                ranked_column_similarity_map = self._column_name_matching(
                    colnames_input, colnames_target)
                all_matches.update(ranked_column_similarity_map)

        if self.use_instances:

            type2cols_input = get_type2columns_map(input)
            type2cols_target = get_type2columns_map(target)

            categorical_colnames_input = type2cols_input['categorical']
            categorical_cols_target = type2cols_target['categorical']

            if len(categorical_colnames_input) > 0 and len(categorical_cols_target) > 0:
                # print("Matching categorical columns")
                ranked_column_similarity_map = self._column_value_matching(
                    input[categorical_colnames_input], target[categorical_cols_target])
                all_matches.update(ranked_column_similarity_map)

        matches = {}
        for col_input, matches_dict in all_matches.items(): 
           for col_target, score in matches_dict.items():
                match = Match(target_df.name, col_target,
                              source_df.name, col_input, score).to_dict
                matches.update(match)

        return matches

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
import tqdm
import pandas as pd
import numpy as np
from .utils import preprocess_string, common_prefix, clean_column_name, get_type2columns_map


class ValueSimilarityRankerTFIDF:
    def __init__(self, topk=10, sim_threshold=0.6):
        self.topk = topk
        self.sim_threshold = sim_threshold

    def _encode(self, col_name, col_values, mode='OnlyValues'):
        if mode == 'OnlyValues':
            return col_values

        if mode == 'ColumnNameAndValue':
            return [col_name + ": " + str(val) for val in col_values]

        if mode == 'ColumnNameAndMultipleValues':
            sorted_values = sorted(col_values)
            result = []
            for i in range(0, len(sorted_values), 4):
                group = sorted_values[i:i + 4]
                group_str = ", ".join(map(str, group))
                result.append(f"{col_name}: {group_str}")
            return result
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _create_column_value_tfidf_dict(self, df):
        col2_val_tfidf = {}

        def process_column(column):
            col_values = df[column].dropna().unique().tolist()
            col_values = self._encode(column, col_values)
            tfidf_matrix = self._get_tfidf_matrix(col_values)
            return column, col_values, tfidf_matrix

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_column, column): column for column in df.columns}

            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                column, col_values, tfidf_matrix = future.result()
                col2_val_tfidf[column] = (col_values, tfidf_matrix)
                
        return col2_val_tfidf

    def _get_tfidf_matrix(self, texts):
        vectorizer = TfidfVectorizer()
        print(texts)
        tfidf_matrix = vectorizer.fit_transform(texts)
        return tfidf_matrix

    def get_type_and_value_based_candidates(self, source_df, target_df):
        candidates = {}

        type2cols_source = get_type2columns_map(source_df)
        type2cols_target = get_type2columns_map(target_df)

        cat_cols_source = type2cols_source['categorical']
        cat_cols_target = type2cols_target['categorical']

        if len(cat_cols_source) == 0 or len(cat_cols_target) == 0:
            return {}

        source_df_cat = source_df[cat_cols_source]
        target_df_cat = target_df[cat_cols_target]

        col2valtfidf_input = self._create_column_value_tfidf_dict(source_df_cat)
        col2valtfidf_target = self._create_column_value_tfidf_dict(target_df_cat)

        all_target_tfidf = []
        all_target_values = []
        target_column_mapping = []

        current_position = 0
        target_column_boundaries = {}

        for target_col, (target_values, target_tfidf) in col2valtfidf_target.items():
            num_values = len(target_values)
            all_target_tfidf.append(target_tfidf)
            all_target_values.extend(target_values)
            target_column_mapping.extend([target_col] * num_values)
            target_column_boundaries[target_col] = (current_position, current_position + num_values)
            current_position += num_values

        all_target_tfidf = np.vstack(all_target_tfidf)

        for input_col, (input_values, input_tfidf) in col2valtfidf_input.items():
            top_k_scores, top_k_indices, similarities = self.compute_tfidf_similarity(
                input_tfidf, all_target_tfidf, self.topk)

            top_k_columns = {target_column_mapping[idx] for row_indices in top_k_indices for idx in row_indices}

            for target_col in list(top_k_columns):
                indx_begin, idx_end = target_column_boundaries[target_col]
                similarities_col = similarities[:, indx_begin:idx_end]

                max_scores = np.max(similarities_col, axis=1)
                score_sum = np.mean(max_scores)

                if score_sum >= self.sim_threshold:
                    candidates[(input_col, target_col)] = score_sum

        return candidates

    def compute_tfidf_similarity(self, source_tfidf, target_tfidf, topk):
        cosine_sim = cosine_similarity(source_tfidf, target_tfidf)
        top_k_scores = np.sort(cosine_sim, axis=1)[:, -topk:]
        top_k_indices = np.argsort(cosine_sim, axis=1)[:, -topk:]
        return top_k_scores, top_k_indices, cosine_sim

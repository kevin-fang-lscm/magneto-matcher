import os
import pandas as pd
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
import faiss
import tqdm
import numpy as np
import re

from valentine.algorithms.base_matcher import BaseMatcher
from valentine.data_sources.base_table import BaseTable
from valentine.algorithms.match import Match

from .indexes import SimilarityIndex
from .datatypes_utils import detect_column_type, is_null, expand_column_name
from .utils import normalize_scores, are_similar_ignore_case_and_punctuation, are_equal_ignore_case_and_punctuation, is_main_theme_similar, hash_string_list

from sentence_transformers import SentenceTransformer
from collections import Counter


os.environ["TOKENIZERS_PARALLELISM"] = "false"

curr_directory = os.path.dirname(os.path.realpath(__file__))
CACHE_DIR = os.path.join(curr_directory, "cache_col_names")


class IndexedSimilarityMatcher(BaseMatcher):
    def __init__(self, model_name='all-MiniLM-L6-v2', top_k: int = 40):

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        self.top_k = top_k

        self.source_df = None
        self.target_df = None

        self.source_categorical_df = None
        self.source_numeric_df = None
        self.source_binary_df = None

        self.source_gene_df = None

        self.target_categorical_df = None
        self.target_numeric_df = None
        self.target_binary_df = None

        self.target_gene_df = None

        self.catdomain_inverted_index = None
        self.catcol2indexes_dict = {}

        self.positions = []

    def _organize_columns_by_type(self, source_df, target_df):

        self.source_df = source_df
        self.target_df = target_df

        source_categorical_columns = []
        source_numeric_columns = []
        source_binary_columns = []

        source_gene_columns = []

        for col in source_df.columns:

            if "gene" in col.lower():
                source_gene_columns.append(col)
            elif detect_column_type(source_df[col]) == 'binary':
                source_binary_columns.append(col)
            elif detect_column_type(source_df[col]) == 'numeric':
                source_numeric_columns.append(col)
            else:
                source_categorical_columns.append(col)

        self.source_categorical_df = source_df[source_categorical_columns]
        self.source_numeric_df = source_df[source_numeric_columns]
        self.source_binary_df = source_df[source_binary_columns]

        self.source_gene_df = source_df[source_gene_columns]

        # print('source_categorical_columns', source_categorical_columns)
        # print('source_numeric_columns', source_numeric_columns)
        # print('source_binary_columns', source_binary_columns)
        # print('source_gene_columns', source_gene_columns)
        # print('\n')

        target_categorical_columns = []
        target_numeric_columns = []
        target_binary_columns = []

        target_gene_columns = []

        for col in target_df.columns:
            if "gene" in col.lower():
                target_gene_columns.append(col)
            elif detect_column_type(target_df[col]) == 'binary':
                target_binary_columns.append(col)
            elif detect_column_type(target_df[col]) == 'numeric':
                target_numeric_columns.append(col)
            else:
                target_categorical_columns.append(col)

        self.target_categorical_df = target_df[target_categorical_columns]
        self.target_numeric_df = target_df[target_numeric_columns]
        self.target_binary_df = target_df[target_binary_columns]
        self.target_gene_df = target_df[target_gene_columns]

    def _get_categorical_candidate_columns(self, column, clean_nulls=True, splitCamelCase=True):

        values = self.source_categorical_df[column].dropna().unique()
        if clean_nulls:
            values = [value for value in values if not is_null(value)]
        if splitCamelCase:
            values = [re.sub('([a-z0-9])([A-Z])', r'\1 \2', value)
                      for value in values]

        candidates_columns = set()

        # adding the column name terms in the search
        column_terms = expand_column_name(column)
        values.extend(column_terms)

        # print(f'Querying for {column} values')
        for value in values:
            results = self.catdomain_inverted_index.query(value, 10)
            # print(f'Querying for {value} ... ')
            for result in results:
                # print('\t', result)
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

    def _handle_categorical_complex(self):

        print('Handling categorical columns')
        self.catdomain_inverted_index = SimilarityIndex(
            self.target_categorical_df)

        cat_matches = {}
        for col in tqdm.tqdm(self.source_categorical_df.columns):

            # if not col.lower()=='Ethnicity_Self_Identify'.lower():
            #     continue

            unique_ratio = len(self.source_categorical_df[col].dropna(
            ).unique()) / len(self.source_categorical_df[col])
            if unique_ratio > 0.80:  # approximate key
                # print(f"Number of unique values in column {col} is more than 99%")
                cat_matches[col] = {"submitter_id": 1.0}
                return cat_matches
                

            candidates_columns = self._get_categorical_candidate_columns(col)

            sorted_candidate_col_max_sim = self._rerank_categorical_candidates(
                col, candidates_columns)

            topk_entries = sorted_candidate_col_max_sim[:self.top_k]

            topk_entries = normalize_scores(topk_entries)

            # re ordering based on column name
            equal_entries = [
                entry for entry in topk_entries if are_equal_ignore_case_and_punctuation(col, entry[0])]
            equal_entries = [(entry[0], 1.0) for entry in equal_entries]

            similar_entries = [
                entry for entry in topk_entries if are_similar_ignore_case_and_punctuation(col, entry[0])]
            similar_entries = [(entry[0], 0.9999) for entry in similar_entries]

            similar_named_entries_out_topk = [
                (target_col, 0.9998) for target_col in self.target_df.columns if are_similar_ignore_case_and_punctuation(col, target_col)]

            # similar_named_entries_out_topk = [
            #     (target_col, 0.9998) for target_col in self.target_df.columns if is_main_theme_similar(col, target_col)]

            non_similar_column_name = [
                entry for entry in topk_entries if entry not in equal_entries]
            non_similar_column_name = [
                entry for entry in non_similar_column_name if entry not in similar_entries]

            topk_entries = {}
            topk_entries.update(equal_entries)
            topk_entries.update(similar_entries)
            topk_entries.update(similar_named_entries_out_topk)
            topk_entries.update(non_similar_column_name)

            # print(f'Column {col} topk entries:')
            # print(topk_entries)

            cat_matches[col] = topk_entries

            # source, target = [
            #     col_pair for col_pair in ground_truth if col_pair[0] == col][0]

            # position = -1
            # if target in topk_entries:
            #     position = list(topk_entries.keys()).index(target)
            # else:
            #     print(f'Column {col} ... {source} -> {target} not found in topk entries')
            #     print(topk_entries)
            #     print(self.source_df[source].unique())
            #     print(self.target_df[target].unique())
            #     print('\n')

            # self.positions.append(position)

        return cat_matches

    def _find_similar_column_names(self, colnames_candidates, colnames_target):

        colname_hash = str(hash_string_list(colnames_target))
        cache_file = os.path.join(CACHE_DIR, self.model_name, colname_hash)

        embeddings = []
        if not os.path.isfile(cache_file + ".npy"):
            # Encode target column names

            for target_colname in tqdm.tqdm(colnames_target, desc="Encoding target columns"):
                embedding = self.model.encode(
                    target_colname, show_progress_bar=False)
                embeddings.append(embedding)

            embeddings = np.array(embeddings).astype('float32')

            # save embeddings on disk
            index_file_dir = os.path.dirname(cache_file)
            os.makedirs(index_file_dir, exist_ok=True)
            np.save(cache_file, embeddings)
        else:
            print("Loading embeddings from disk ...")
            embeddings = np.load(cache_file + ".npy")

        print("Building FAISS index ...")
        # Normalize embeddings and build index
        faiss.normalize_L2(embeddings)
        # Create FAISS index
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)

        matches = {}
        for colname in tqdm.tqdm(colnames_candidates, desc="Finding matches"):
            query_embedding = self.model.encode(
                colname, show_progress_bar=False).astype('float32')
            faiss.normalize_L2(query_embedding.reshape(1, -1))

            D, I = index.search(query_embedding.reshape(1, -1), self.top_k)

            colname_matches = {}
            for i, idx in enumerate(I[0]):
                colname_matches[colnames_target[idx]] = float(D[0][i])

            matches[colname] = colname_matches

        return matches

    # def _handle_categorical(self, ground_truth):

    #     print('Handling categorical columns')

    #     cat_matches = self._find_similar_column_names(
    #         self.source_categorical_df.columns, self.target_categorical_df.columns)

    #     for col in self.source_categorical_df.columns:
    #         source, target = [
    #             col_pair for col_pair in ground_truth if col_pair[0] == col][0]
    #         topk_entries = cat_matches[col]
    #         position = -1
    #         if target in topk_entries:
    #             position = list(topk_entries.keys()).index(target)
    #         else:
    #             print(f'Column {source}->{target} not found in topk entries')
    #             # print(topk_entries)
    #             is_in = target in self.target_categorical_df.columns
    #             print(f'Column {target} in target: {is_in}')

    #         self.positions.append(position)

    def _handle_numerical(self):
        # print('Handling numerical columns')

        num_matches = self._find_similar_column_names(
            self.source_numeric_df.columns, self.target_numeric_df.columns)

        # for col in self.source_numeric_df.columns:
        #     source, target = [
        #         col_pair for col_pair in ground_truth if col_pair[0] == col][0]
        #     topk_entries = num_matches[col]
        #     position = -1
        #     if target in topk_entries:
        #         position = list(topk_entries.keys()).index(target)
        #     # else:
        #     #     print(f'Column {source}->{target} not found in topk entries')
        #     #     # print(topk_entries)

        #     #     is_in = target in self.target_numeric_df.columns
        #     #     print(f'Column {target} in target: {is_in}')

        #     self.positions.append(position)

        return num_matches

    def _handle_binary(self):
        # print('Handling binary columns')
        binary_matches = self._find_similar_column_names(
            self.source_binary_df.columns, self.target_binary_df.columns)

        # for col in self.source_binary_df.columns:
        #     source, target = [
        #         col_pair for col_pair in ground_truth if col_pair[0] == col][0]
        #     topk_entries = binary_matches[col]
        #     position = -1
        #     if target in topk_entries:
        #         position = list(topk_entries.keys()).index(target)
        # else:
        #     print(
        #         f'Column {source}->{target} not found in topk entries')

        #     is_in = target in self.target_binary_df.columns
        #     print(f'Column {target} in target: {is_in}')

        # self.positions.append(position)

        return binary_matches

    def get_matches(self,
                    source_input: BaseTable,
                    target_input: BaseTable) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:

        source_df = source_input.get_df()

        target_df = target_input.get_df()

        self._organize_columns_by_type(source_df, target_df)

        cat_matches = self._handle_categorical_complex()

        num_matches = self._handle_numerical()

        binary_matches = self._handle_binary()


        matches = {}
        matches.update(cat_matches)
        matches.update(num_matches)
        matches.update(binary_matches)

        # print('Matches: ', matches)

        final_matches = { }

        for col_input, matches_col in matches.items():
            for col_target, score in matches_col.items():
                match = Match(target_input.name, col_target,
                        source_input.name, col_input,
                        score).to_dict
                final_matches.update(match)

        # print('Final Matches: ', final_matches)
            
        # Remove the pairs with zero similarity
        matches = {k: v for k, v in final_matches.items() if v > 0.0}

        return matches


# def get_dataframes(file):
#     # gdc_cat_path = './data/gdc/gdc_cat_ingt.csv'
#     # gdc_cat_path = './data/gdc/gdc_cat.csv'
#     gdc_cat_path = './data/gdc/gdc_num_cat.csv'

#     df_gdc = pd.read_csv(gdc_cat_path, encoding='utf-8', engine='python')

#     df_input = pd.read_csv(f'./data/gdc/source-tables/{file}')

#     gt_df = pd.read_csv(
#         f'./data/gdc/ground-truth/{file}', encoding='utf-8', engine='python')
#     ground_truth = list(gt_df.itertuples(index=False, name=None))

#     gt_gdc_cols = set([col[1] for col in ground_truth])
#     gt_gdc_cols = set(gt_gdc_cols).intersection(df_gdc.columns)
#     gt_gdc_cols_in_input_df = [col_pair[0]
#                                for col_pair in ground_truth if col_pair[1] in gt_gdc_cols]

#     df_input = df_input[gt_gdc_cols_in_input_df]

#     return df_input, df_gdc, ground_truth


# def mean_reciprocal_rank(positions):
#     reciprocal_ranks = []
#     for position in positions:
#         if position != -1:
#             reciprocal_ranks.append(1 / (position + 1))
#     if reciprocal_ranks:
#         return sum(reciprocal_ranks) / len(positions)
#     else:
#         return 0.0


# if __name__ == '__main__':

#     files = ['Dou.csv', 'Krug.csv', 'Clark.csv',  'Vasaikar.csv',
#              'Wang.csv', 'Satpathy.csv', 'Cao.csv', 'Huang.csv', 'Gilette.csv']
#     # file = files[-1]

#     files = ['Clark.csv']

#     positions_global = []
#     for file in files:
#         print("File: ", file)

#         df_input, df_gdc, ground_truth = get_dataframes(file)

#         matcher = IndexedSimilarityMatcher()
#         positions = matcher.get_matches(df_input, df_gdc, ground_truth)

#         positions_global.extend(positions)

#         print('\n')

#     print(positions_global)

#     num_of_non_matches = positions_global.count(-1)

#     print(
#         f'Number of non-matches: {num_of_non_matches} out of {len(positions_global)}')

#     mrr = mean_reciprocal_rank(positions_global)
#     print(f"Mean Reciprocal Rank: {mrr}")

#     number_frequency = Counter(positions_global)
#     print("Frequency of each number:")
#     for number, frequency in number_frequency.items():
#         print(f"{number}: {frequency}")

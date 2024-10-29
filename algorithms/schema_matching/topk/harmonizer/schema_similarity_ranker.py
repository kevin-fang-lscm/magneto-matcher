
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
from .utils import preprocess_string, common_prefix, clean_column_name, remove_invalid_characters
from .embedding_utils import compute_cosine_similarity, compute_cosine_similarity_simple

from fuzzywuzzy import fuzz

class SchemaSimilarityRanker:
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2',
                 topk=5, embedding_threshold=0.5, alignment_threshold=0.95, fuzzy_similarity_threshold=0.4):

        if model_name is None:
            #model_name = 'sentence-transformers/all-mpnet-base-v2'
            model_name = 'sentence-transformers/all-mpnet-base-v2'
            
        self.model_name = model_name
        print(f"Loading model {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.topk = topk
        self.embedding_threshold = embedding_threshold

        self.alignment_threshold = alignment_threshold
        self.fuzzy_similarity_threshold = fuzzy_similarity_threshold

    def alignment_score_consecutive(self, str1, str2, max_distance=2, size_ratio_threshold=3):
        s1 = str1
        s2 = str2
        # Preprocess strings (assuming this function exists)
        str1 = preprocess_string(str1)
        str2 = preprocess_string(str2)

        # Determine shorter and longer strings
        if len(str1) <= len(str2):
            shorter, longer = str1, str2
        else:
            shorter, longer = str2, str1

        # Early exit if strings have disproportionate lengths
        if len(longer) > len(shorter) * size_ratio_threshold:
            return 0

        matches = 0
        last_index = -1

        # Find matches for each letter in the shorter string
        for char in shorter:
            for i in range(last_index + 1, len(longer)):
                if longer[i] == char:
                    # Check if the distance between the current match and the last one is <= max_distance
                    if last_index == -1 or (i - last_index) <= max_distance:
                        matches += 1
                        last_index = i
                        break
                    else:
                        # If the distance is greater than max_distance, stop the inner loop
                        break

        # Calculate score
        score = matches / len(shorter) if len(shorter) > 0 else 0

        return score

    def fuzzy_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity using fuzzy matching (Levenshtein distance)."""
        return fuzz.ratio(s1, s2) / 100.0  # Normalize the score to a range [0, 1]

    def get_str_similarity_candidates(self, source_column_names, target_column_names):

        prefix_source = common_prefix(list(source_column_names))
        prefix_target = common_prefix(list(target_column_names))

        candidates = {}
        for source_col in source_column_names:
            prep_source_col = source_col.replace(prefix_source, "")

            for target_col in target_column_names:
                prep_target_col = target_col.replace(prefix_target, "")


                

                alignment_score = self.alignment_score_consecutive(
                    prep_source_col, prep_target_col)

                if alignment_score >= self.alignment_threshold:
                    candidates[(source_col, target_col)] = alignment_score

                name_similarity = self.fuzzy_similarity(
                    prep_source_col, prep_target_col)

                if name_similarity >= self.fuzzy_similarity_threshold:
                    candidates[(source_col, target_col)] = name_similarity

        return candidates

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

    def get_embedding_similarity_candidates(self, source_column_names, target_column_names):

        input_colnames_dict = {clean_column_name(
            col): col for col in source_column_names}
        target_colnames_dict = {clean_column_name(
            col): col for col in target_column_names}

        cleaned_input_colnames = list(input_colnames_dict.keys())
        cleaned_target_colnames = list(target_colnames_dict.keys())

        embeddings_input = self._get_embeddings(cleaned_input_colnames)
        embeddings_target = self._get_embeddings(cleaned_target_colnames)

        top_k = min(self.topk, len(cleaned_target_colnames))
        topk_similarity, topk_indices = compute_cosine_similarity_simple(
            embeddings_input, embeddings_target, top_k)

        candidates = {}
        for i, cleaned_input_col in enumerate(cleaned_input_colnames):
            original_input_col = input_colnames_dict[cleaned_input_col]

            for j in range(top_k):
                cleaned_target_col = cleaned_target_colnames[topk_indices[i, j]]
                original_target_col = target_colnames_dict[cleaned_target_col]
                similarity = topk_similarity[i, j].item()

                if similarity >= self.embedding_threshold:

                    candidates[(original_input_col,
                                original_target_col)] = similarity
        return candidates

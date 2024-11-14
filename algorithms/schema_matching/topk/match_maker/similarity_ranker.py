
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from fuzzywuzzy import fuzz

from .utils import preprocess_string, common_prefix,  get_samples, detect_column_type
from .embedding_utils import compute_cosine_similarity_simple


class SimilarityRanker:
    def __init__(self, fine_tune_path=None, topk=20, embedding_threshold=0.65, alignment_threshold=0.95, fuzzy_similarity_threshold=0.4):

        self.device = torch.device("cpu")
        # print("Using CPU for model inference.")
        
        if fine_tune_path is None:
            self.model_name = 'sentence-transformers/all-mpnet-base-v2'
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            print(f"Loaded ZeroShot Model on {self.device}")
        else:
            # Load the fine-tuned SentenceTransformer model on the CPU
            model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
            model.load_state_dict(torch.load(fine_tune_path, map_location=self.device, weights_only=True))
            # model.load_state_dict(torch.load(fine_tune_path, map_location=self.device))

            # Access the first module's transformer model (AutoModel) and tokenizer
            transformer_model = model._first_module().auto_model
            transformer_tokenizer = model._first_module().tokenizer

            # Set the tokenizer and model to use the CPU
            self.model_name = transformer_model.config._name_or_path
            self.tokenizer = transformer_tokenizer
            self.model = transformer_model.to(self.device)
            
            print(f"Loaded FineTuned Model {fine_tune_path} on {self.device}")
        # else:
        #     # Load the fine-tuned SentenceTransformer model using the SentenceTransformer.load method
        #     model = SentenceTransformer(fine_tune_path)

        #     # Access the model's underlying transformer model and tokenizer
        #     transformer_model = model._first_module().auto_model
        #     transformer_tokenizer = model._first_module().tokenizer

        #     # Set the tokenizer and model to use the CPU
        #     self.model_name = transformer_model.config._name_or_path
        #     self.tokenizer = transformer_tokenizer
        #     self.model = transformer_model.to(self.device)
            
        #     print(f"Loaded FineTuned Model {fine_tune_path} on {self.device}")

        
        self.embedding_threshold = embedding_threshold

        self.topk = topk

        self.alignment_threshold = alignment_threshold
        self.fuzzy_similarity_threshold = fuzzy_similarity_threshold

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

    def encode(self, df, col):

        header = col
        data_type = detect_column_type(df[col])
        tokens = get_samples(df[col])

        text = (
            self.tokenizer.cls_token
            + "Column: " + header
            + self.tokenizer.sep_token
            + "Type: " + data_type
            + self.tokenizer.sep_token
            + "Values: " + self.tokenizer.sep_token.join(tokens)
            + self.tokenizer.sep_token
            + self.tokenizer.eos_token  # End-of-sequence token
        )
        # print(text)

        return text

    def get_embedding_similarity_candidates(self, source_df, target_df):

        input_col_repr_dict = {self.encode(source_df,
                                           col): col for col in source_df.columns}
        target_col_repr_dict = {self.encode(target_df,
                                            col): col for col in target_df.columns}

        cleaned_input_col_repr = list(input_col_repr_dict.keys())
        cleaned_target_col_repr = list(target_col_repr_dict.keys())

        embeddings_input = self._get_embeddings(cleaned_input_col_repr)
        embeddings_target = self._get_embeddings(cleaned_target_col_repr)

        top_k = min(self.topk, len(cleaned_target_col_repr))
        topk_similarity, topk_indices = compute_cosine_similarity_simple(
            embeddings_input, embeddings_target, top_k)

        candidates = {}

        for i, cleaned_input_col in enumerate(cleaned_input_col_repr):
            original_input_col = input_col_repr_dict[cleaned_input_col]

            for j in range(top_k):
                cleaned_target_col = cleaned_target_col_repr[topk_indices[i, j]]
                original_target_col = target_col_repr_dict[cleaned_target_col]
                similarity = topk_similarity[i, j].item()

                if similarity >= self.embedding_threshold:

                    candidates[(original_input_col,
                                original_target_col)] = similarity

        return candidates

    def alignment_score_consecutive(self, str1, str2, max_distance=2, size_ratio_threshold=2):
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
        # Normalize the score to a range [0, 1]
        return fuzz.ratio(s1, s2) / 100.0


import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from fuzzywuzzy import fuzz

from .utils import get_samples, detect_column_type
from .embedding_utils import compute_cosine_similarity_simple
from .column_encoder import ColumnEncoder

DEFAULT_MODELS = ["sentence-transformers/all-mpnet-base-v2"]


class EmbeddingMatcher:
    def __init__(self, params):
        self.params = params
        self.topk = params['topk']
        self.embedding_threshold = params['embedding_threshold']

        self.device = torch.device("cpu")  # todo support gpu

        self.model_name = params['embedding_model']
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

        print(f"Loaded ZeroShot Model on {self.device}")

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

    # def encode(self, df, col):

    #     header = col
    #     data_type = detect_column_type(df[col])
    #     tokens = get_samples(df[col])

    #     text = (
    #         self.tokenizer.cls_token
    #         + "Column: " + header
    #         + self.tokenizer.sep_token
    #         + "Type: " + data_type
    #         + self.tokenizer.sep_token
    #         + "Values: " + self.tokenizer.sep_token.join(tokens)
    #         + self.tokenizer.sep_token
    #         + self.tokenizer.eos_token  # End-of-sequence token
    #     )

    #     return text

    def get_embedding_similarity_candidates(self, source_df, target_df):

        encoder = ColumnEncoder(
            self.tokenizer, mode=self.params['encoding_mode'])

        input_col_repr_dict = {encoder.encode(source_df,
                                              col): col for col in source_df.columns}
        target_col_repr_dict = {encoder.encode(target_df,
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

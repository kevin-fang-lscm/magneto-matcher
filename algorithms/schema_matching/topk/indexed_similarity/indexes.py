import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import tqdm
import bisect
from datatypes_utils import clean_null_cells
from utils import hash_dataframe

curr_directory = os.path.dirname(os.path.realpath(__file__))
CACHE_DIR = os.path.join(curr_directory, "cache")


class SimilarityIndex:
    def __init__(self, df, model_name="dmis-lab/biobert-v1.1", clean_nulls=True):
        if clean_nulls:
            self.df = clean_null_cells(df)
        else:
            self.df = df

        df_hash = str(hash_dataframe(self.df))
        index_file = os.path.join(CACHE_DIR, model_name, df_hash)

        self.inverted_index = self._prepare_inverted_index_entries()
        self.sorted_index_keys = sorted(self.inverted_index.keys())
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        if not os.path.isfile(index_file + ".npy"):
            self.embeddings = self._embed_entries(self.sorted_index_keys)

            # save embeddings on disk
            index_file_dir = os.path.dirname(index_file)
            os.makedirs(index_file_dir, exist_ok=True)
            np.save(index_file, self.embeddings)

            self.index = self._build_faiss_index(self.embeddings)
        else:
            print("Loading embeddings from disk ...")
            self.embeddings = np.load(index_file + ".npy")
            self.index = self._build_faiss_index(self.embeddings)

    def _prepare_inverted_index_entries(self):
        inverted_index = {}
        for column in self.df.columns:
            for value in self.df[column].dropna():
                if value not in inverted_index:
                    inverted_index[value] = []
                inverted_index[value].append(column)
        return inverted_index

    def encode_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.squeeze().numpy()

    def _embed_entries(self, entries):
        print("Encoding values ...")
        embeddings = []
        for entry in tqdm.tqdm(entries, desc="Encoding texts"):
            embeddings.append(self.encode_text(entry))
        embeddings_np = np.array(embeddings).astype('float32')
        return embeddings_np

    def _build_faiss_index(self, embeddings):
        print("Building faiss index ...")
        faiss.normalize_L2(embeddings)
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        print("Done.")
        return index

    def query(self, query_text, k=40):
        query_embedding = self.encode_text(query_text).astype('float32')
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        D, I = self.index.search(query_embedding.reshape(1, -1), k)
        results = []
        for i, idx in enumerate(I[0]):
            result = {
                'rank': i + 1,
                'index': idx,
                'similar_value': self.sorted_index_keys[idx],
                'columns': self.inverted_index[self.sorted_index_keys[idx]],
                'similarity': D[0][i]
            }
            results.append(result)
        return results

    def get_col_index(self, col):
        col_values = self.df[col].dropna().unique()

        col_embeddings = []
        for value in col_values:
            idx = bisect.bisect_left(self.sorted_index_keys, value)
            if idx < len(self.sorted_index_keys) and self.sorted_index_keys[idx] == value:
                col_embeddings.append(self.embeddings[idx])
            else:
                raise ValueError(f"Value {value} not found in index")

        col_embeddings_np = np.array(col_embeddings).astype('float32')
        faiss.normalize_L2(col_embeddings_np)

        d = col_embeddings_np.shape[1]
        col_index = faiss.IndexFlatIP(d)
        col_index.add(col_embeddings_np)

        return (col_values, col_index)

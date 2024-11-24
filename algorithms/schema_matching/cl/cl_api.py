import hashlib
from joblib import Memory
import os
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from .cl_models import (
    BarlowTwinsSimCLR,
)
from .cl_pretrained_dataset import (
    PretrainTableDataset,
)
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

dir_path = os.path.dirname(os.path.realpath(__file__))
GDC_TABLE_PATH = os.path.join(dir_path, "resource/gdc_table.csv")

MODEL_PATH = os.path.join(dir_path, "resource/model_20_1.pt")

DEFAULT_CACHE_PATH = os.getenv(
    "XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")
)

location = DEFAULT_CACHE_PATH
memory = Memory(location, verbose=0)


def hash_dataframe(df: pd.DataFrame) -> str:

    hash_object = hashlib.sha256()

    columns_string = ",".join(df.columns) + "\n"
    hash_object.update(columns_string.encode())

    for row in df.itertuples(index=False, name=None):
        row_string = ",".join(map(str, row)) + "\n"
        hash_object.update(row_string.encode())

    return hash_object.hexdigest()


class ContrastiveLearningAPI:
    def __init__(self, model_path=MODEL_PATH, top_k=10, batch_size=128):
        self.model_path = model_path
        self.unlabeled = PretrainTableDataset()
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_checkpoint()
        self.top_k = top_k

    def load_checkpoint(self, lm="roberta"):
        ckpt = torch.load(self.model_path, map_location=torch.device("cpu"))
        scale_loss = 0.1
        lambd = 3.9
        model = BarlowTwinsSimCLR(scale_loss, lambd, device=self.device, lm=lm)
        model = model.to(self.device)
        model.load_state_dict(ckpt["model"])

        return model

    def get_recommendations(
        self, table: pd.DataFrame, target: Optional[Union[str, pd.DataFrame]] = None
    ):
        if target is None or (isinstance(target, str) and target == "gdc"):
            gdc_ds = pd.read_csv(GDC_TABLE_PATH)
        elif isinstance(target, pd.DataFrame):
            gdc_ds = target
        else:
            raise ValueError("Target must be a DataFrame or 'gdc'")

        l_features = self._load_table_tokens(table)

        # df_hash = hash_dataframe(gdc_ds)
        # df_hash_file = os.path.join(DEFAULT_CACHE_PATH, df_hash)
        # r_features = None

        r_features = self._load_table_tokens(gdc_ds)
        # with open(df_hash_file, "w") as file:
        #     file.write('\n'.join([','.join(map(str, vec))
        #                 for vec in r_features]))

        # if not os.path.isfile(df_hash_file):

        #     r_features = self._load_table_tokens(gdc_ds)
        #     with open(df_hash_file, "w") as file:
        #         file.write('\n'.join([','.join(map(str, vec))
        #                    for vec in r_features]))
        # else:
        #     try:
        #         with open(df_hash_file, "r") as file:
        #             r_features = [[float(val) for val in vec.split(',')]
        #                           for vec in file.read().split('\n')]
        #             if len(r_features) != len(gdc_ds.columns):
        #                 raise Exception("Hash file corrupted")
        #     except:
        #         r_features = self._load_table_tokens(gdc_ds)
        #         with open(df_hash_file, "w") as file:
        #             file.write(
        #                 '\n'.join([','.join(map(str, vec)) for vec in r_features]))

        # cached_cosine_similarity = memory.cache(cosine_similarity)

        cosine_sim = cosine_similarity(l_features, r_features)
        # cosine_sim = cached_cosine_similarity(l_features, r_features)

        # print(f"l_features - {len(l_features)}:{l_features[0].shape}\nr-feature - {len(r_features)}:{r_features[0].shape}\nCosine - {cosine_sim.shape}")

        top_k_results = []
        l_column_ids = table.columns
        gt_column_ids = gdc_ds.columns

        for index, similarities in enumerate(cosine_sim):
            top_k_indices = np.argsort(similarities)[::-1][: self.top_k]
            top_k_column_names = [gt_column_ids[i] for i in top_k_indices]
            top_k_similarities = [str(round(similarities[i], 4)) for i in top_k_indices]
            top_k_columns = list(zip(top_k_column_names, top_k_similarities))
            result = {
                "Candidate column": l_column_ids[index],
                "Top k columns": top_k_columns,
            }
            top_k_results.append(result)
        recommendations = self._extract_recommendations_from_top_k(top_k_results)
        return recommendations, top_k_results

    def _extract_recommendations_from_top_k(self, top_k_results):
        recommendations = set()
        for result in top_k_results:
            for name, _ in result["Top k columns"]:
                recommendations.add(name)
        return list(recommendations)

    def _sample_to_15_rows(self, table: pd.DataFrame):
        if len(table) > 15:
            unique_rows = table.drop_duplicates()
            num_unique_rows = len(unique_rows)
            if num_unique_rows <= 15:
                needed_rows = 15 - num_unique_rows
                additional_rows = table[~table.index.isin(unique_rows.index)].sample(
                    n=needed_rows, replace=True, random_state=1
                )
                table = pd.concat([unique_rows, additional_rows])
            else:
                table = unique_rows.sample(n=15, random_state=1)
        return table

    def _load_table_tokens(self, table: pd.DataFrame):
        tables = []
        for i, column in enumerate(table.columns):
            curr_table = pd.DataFrame(table[column])
            curr_table = self._sample_to_15_rows(curr_table)
            tables.append(curr_table)
        vectors = self._inference_on_tables(tables)
        print(f"Table features extracted from {len(table.columns)} columns")
        return [vec[-1] for vec in vectors]

    def _inference_on_tables(self, tables: List[pd.DataFrame]):
        total = len(tables)
        batch = []
        results = []

        for tid, table in tqdm(enumerate(tables), total=total):
            x, _ = self.unlabeled._tokenize(table)
            batch.append((x, x, []))

            if tid == total - 1 or len(batch) == self.batch_size:
                with torch.no_grad():
                    x, _, _ = self.unlabeled.pad(batch)
                    column_vectors = self.model.inference(x)
                    ptr = 0
                    for xi in x:
                        current = []
                        for token_id in xi:
                            if token_id == self.unlabeled.tokenizer.cls_token_id:
                                current.append(column_vectors[ptr].cpu().numpy())
                                ptr += 1
                        results.append(current)
                batch.clear()
        return results

    def get_type(self):
        return "ContrastiveLearning"

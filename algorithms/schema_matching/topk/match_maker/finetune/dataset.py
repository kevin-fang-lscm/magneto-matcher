from algorithms.schema_matching.topk.match_maker.utils import detect_column_type, clean_element
from train_utils import sentence_transformer_map
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import Dataset

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class CustomDataset(Dataset):
    def __init__(
        self,
        data,
        model_type="mpnet",
        serialization="header_values_verbose",
        augmentation="exact_semantic",
    ):
        self.serialization = serialization
        self.tokenizer = AutoTokenizer.from_pretrained(
            sentence_transformer_map[model_type]
        )
        self.labels = []
        self.items = self._initialize_items(data, augmentation)

    def _initialize_items(self, data, augmentation):
        items = []
        class_id = 0

        for _, categories in data.items():
            for aug_type, columns in categories.items():
                if aug_type in augmentation or aug_type == "original":
                    for column_name, values in columns.items():
                        processed_column_name = (
                            column_name.rsplit("_", 1)[0]
                            if aug_type == "exact"
                            else column_name
                        )
                        # processed_column_name = clean_element(processed_column_name)
                        values = [clean_element(str(value))
                                  for value in values]
                        items.append((processed_column_name, values, class_id))
                        self.labels.append(class_id)
            class_id += 1

        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        key, values, class_id = self.items[idx]
        text = self._serialize(key, values)
        return text, class_id

    # def _serialize(self, header, values):
    #     if values:
    #         col = pd.DataFrame({header: values})[header]
    #         data_type = detect_column_type(pd.DataFrame({header: values})[header])
    #     else:
    #         data_type = "unknown"
    #     serialization = {
    #         "header_values_default": f"{self.tokenizer.cls_token}{header}{self.tokenizer.sep_token}{data_type}{self.tokenizer.sep_token}{','.join(map(str, values))}",
    #         "header_values_prefix": f"{self.tokenizer.cls_token}header:{header}{self.tokenizer.sep_token}datatype:{data_type}{self.tokenizer.sep_token}values:{', '.join(map(str, values))}",
    #     }
    #     return serialization[self.serialization]
    def _serialize(self, header, values):
        if values:
            col = pd.DataFrame({header: values})[header]
            data_type = detect_column_type(col)
        else:
            data_type = "unknown"
            values = []

        serialization = {
            "header_values_default": f"{self.tokenizer.cls_token}{header}{self.tokenizer.sep_token}{data_type}{self.tokenizer.sep_token}{','.join(map(str, values))}",
            "header_values_prefix": f"{self.tokenizer.cls_token}header:{header}. {self.tokenizer.sep_token}datatype:{data_type}{self.tokenizer.sep_token}values:{', '.join(map(str, values))}",
            "header_values_repeat": f"{self.tokenizer.cls_token}{self.tokenizer.sep_token.join([header] * 5)}{self.tokenizer.sep_token}{data_type}{self.tokenizer.sep_token}{','.join(map(str, values))}",
            "header_only": f"{self.tokenizer.cls_token}{header}{self.tokenizer.eos_token}",
            "header_values_simple": f"{self.tokenizer.cls_token}Column: {header}{self.tokenizer.sep_token}Values: {','.join(map(str, values))}{self.tokenizer.sep_token}{self.tokenizer.eos_token}",
        }

        serialization["header_values_verbose"] = (
            self.tokenizer.cls_token
            + "Column: " + header
            + self.tokenizer.sep_token
            + "Type: " + data_type
            + self.tokenizer.sep_token
            + "Values: " + self.tokenizer.sep_token.join(map(str, values))
            + self.tokenizer.sep_token
            + self.tokenizer.eos_token
        )

        return serialization[self.serialization]

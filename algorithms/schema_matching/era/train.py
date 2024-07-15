from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader, Dataset
import os
import sys
import random
import pandas as pd
import torch

os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# project_path = os.path.dirname(os.path.dirname(
#     os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.join(project_path, 'algorithms', 'schema_matching', 'era'))

from table2text import Dataset2Text, ComplexColumnSummary

current_dir = os.getcwd()

# TRAINDATA_DIR = os.path.join(current_dir,  'data', 'table_gpt', 'train')
# MODEL_PATH = os.path.join(current_dir,  'model', 'fine_tablegpt')

TRAINDATA_DIR = os.path.join(current_dir,  'data', 'gdc', 'train')
MODEL_PATH = os.path.join(current_dir,  'model', 'fine_gdc')

TRAIN_MINI_BATCH_SIZE = 16

class Dataset2TextTrainning(Dataset):
    def __init__(self):
        self.data = []

    def add_example(self, sentence1, sentence2):
        self.data.append(InputExample(texts=[sentence1, sentence2]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_training_dataset(directory):
    dataTransformer = Dataset2Text(
        num_context_columns=0, num_context_rows=0, col_summary_impl=ComplexColumnSummary())

    dataset = Dataset2TextTrainning()

    folder_list = [f.path for f in os.scandir(directory) if f.is_dir()]
    random.shuffle(folder_list)

    for folder_path in folder_list:
        df1 = pd.read_csv(os.path.join(folder_path, 'table_lhs.csv'))
        _, col2TextLhs = dataTransformer.transform(df1)

        df2 = pd.read_csv(os.path.join(folder_path, 'table_rhs.csv'))
        _, col2TextRhs = dataTransformer.transform(df2)

        gt_df = pd.read_csv(os.path.join(folder_path, 'gt.csv'))
        gt_df = gt_df.dropna(subset=['lhs_col', 'rhs_col'])
        ground_truth_tuples = list(gt_df.itertuples(index=False, name=None))

        for gtTuple in ground_truth_tuples:
            if gtTuple[0] not in col2TextLhs or gtTuple[1] not in col2TextRhs:
                continue

            sentence1 = col2TextLhs[gtTuple[0]]
            sentence2 = col2TextRhs[gtTuple[1]]
            dataset.add_example(sentence1, sentence2)

    return dataset


def train_model(directory, model_path):
    dataset = get_training_dataset(directory)
    print("Training data size: ", len(dataset))

    train_dataloader = DataLoader(dataset, shuffle=True, batch_size=TRAIN_MINI_BATCH_SIZE, pin_memory=True)
    model = SentenceTransformer('all-mpnet-base-v2')

    
    loss = losses.MultipleNegativesRankingLoss(model)

    # Detect the device to use (MPS for Mac, CUDA for GPU, CPU as fallback)
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"Using device: {device}")
    model.to(device)

    
    model.fit(
        train_objectives=[(train_dataloader, loss)],
        epochs=10,  # Set the number of epochs
        warmup_steps=100,  # Number of warmup steps for learning rate scheduler
        output_path=MODEL_PATH  # Directory to save the model
    )

if __name__ == "__main__":
    train_model(TRAINDATA_DIR, MODEL_PATH)

import re
from collections import Counter
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer


class ColumnDataset(Dataset):
    def __init__(self, column_pairs, tokenizer, max_length=128):
        self.column_pairs = column_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.column_pairs)

    def __getitem__(self, idx):
        col1, col2, label = self.column_pairs[idx]

        # Create descriptive text for each column
        text1 = self.format_column_text(col1)
        text2 = self.format_column_text(col2)

        # Tokenize
        encoding = self.tokenizer(
            text1,
            text2,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Convert to appropriate format
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
        }

    def format_column_text(self, column: Tuple[str, List]) -> str:
        """Format column information into descriptive text."""
        name, values = column

        # Get basic statistics
        unique_vals = len(set(values))
        total_vals = len(values)

        # Detect if binary
        is_binary = unique_vals <= 2

        # Create value distribution
        value_counts = Counter(values)
        dist_str = " ".join([f"{k}:{v}" for k, v in value_counts.most_common(3)])

        # Create descriptive text
        text = (
            f"Column: {name} "
            f"Values: {dist_str} "
            f"Unique: {unique_vals}/{total_vals} "
            f"Binary: {is_binary}"
        )

        return text


class SemanticColumnMatcher:
    def __init__(self, model_path: str = None):
        """
        Initialize the semantic column matcher.

        Args:
            model_path: Path to saved model, if None uses pretrained DistilBERT
        """
        # Use DistilBERT for better efficiency while maintaining good performance
        self.model_name = "distilbert-base-uncased"
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)

        if model_path:
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        else:
            self.model = DistilBertForSequenceClassification.from_pretrained(
                self.model_name, num_labels=2
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def preprocess_column(self, column: Tuple[str, List]) -> Tuple[str, List]:
        """Preprocess column name and values."""
        name, values = column

        # Clean column name
        name = re.sub(r"[.\s-]", "_", name.lower())
        name = re.sub(r"[^a-z0-9_]", "", name)

        # Convert values to consistent format
        values = [str(v).lower() for v in values]

        return (name, values)

    def get_value_statistics(self, values: List) -> Dict:
        """Calculate statistical features of column values."""
        value_counts = Counter(values)
        total = len(values)
        unique = len(value_counts)

        return {
            "unique_ratio": unique / total if total > 0 else 0,
            "most_common": list(value_counts.most_common(3)),
            "is_binary": unique <= 2,
        }

    @torch.no_grad()
    def predict(
        self, column1: Tuple[str, List], column2: Tuple[str, List]
    ) -> Dict[str, Union[bool, float, dict]]:
        """
        Predict if two columns match using semantic similarity.

        Args:
            column1: Tuple of (column_name, values)
            column2: Tuple of (column_name, values)

        Returns:
            Dictionary with match prediction and confidence scores
        """
        # Preprocess columns
        col1_clean = self.preprocess_column(column1)
        col2_clean = self.preprocess_column(column2)

        # Create dataset
        dataset = ColumnDataset(
            [(col1_clean, col2_clean, 0)],  # Label doesn't matter for prediction
            self.tokenizer,
        )

        # Get model input
        batch = dataset[0]
        input_ids = batch["input_ids"].unsqueeze(0).to(self.device)
        attention_mask = batch["attention_mask"].unsqueeze(0).to(self.device)

        # Model inference
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Get probabilities
        probs = torch.softmax(outputs.logits, dim=1)
        match_probability = probs[0][1].item()

        # Get value statistics for additional context
        stats1 = self.get_value_statistics(col1_clean[1])
        stats2 = self.get_value_statistics(col2_clean[1])

        return {
            "is_match": match_probability >= 0.5,
            "confidence": match_probability,
            "details": {
                "column1_stats": stats1,
                "column2_stats": stats2,
                "model_logits": outputs.logits[0].tolist(),
            },
        }

    def train(
        self,
        train_data: List[Tuple],
        val_data: List[Tuple] = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
    ):
        """
        Train the model on column matching data.

        Args:
            train_data: List of (column1, column2, label) tuples
            val_data: Optional validation data
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
        """
        train_dataset = ColumnDataset(train_data, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if val_data:
            val_dataset = ColumnDataset(val_data, self.tokenizer)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")


# Example usage
if __name__ == "__main__":
    # Initialize matcher
    matcher = SemanticColumnMatcher()

    # Example columns
    test_pairs = [
        (
            ("TP53.Mutation.Status", [0, 0, 1, 0, 1]),
            ("mutation_status", [False, False, True, False, True]),
        ),
        (
            ("patient_age", [25, 30, 45, 50, 35]),
            ("age_at_diagnosis", [25, 30, 45, 50, 35]),
        ),
    ]

    # Make predictions
    for idx, (col1, col2) in enumerate(test_pairs, 1):
        result = matcher.predict(col1, col2)
        print(f"\nTest Pair {idx}:")
        print(f"Column 1: {col1[0]}")
        print(f"Column 2: {col2[0]}")
        print(f"Match: {result['is_match']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print("Details:", result["details"])

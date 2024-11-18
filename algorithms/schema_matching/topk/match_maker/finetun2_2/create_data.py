# ... (previous imports remain the same)
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch
from typing import List, Tuple, Dict
import random
from collections import defaultdict
from datetime import datetime, timedelta
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

class ColumnSimilarityTrainer:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
        
    def train(self, 
              examples: List[InputExample], 
              batch_size: int = 16,
              epochs: int = 3,
              learning_rate: float = 2e-5,
              warmup_steps: int = 50,
              output_path: str = "column_similarity_model"):
        """Train the model on the prepared examples."""
        
        if not examples:
            raise ValueError("No training examples provided!")
            
        # Create data loader
        train_dataloader = DataLoader(examples, batch_size=batch_size, shuffle=True)
        
        # Calculate total steps
        num_training_steps = len(train_dataloader) * epochs
        
        # Initialize optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, correct_bias=False)
        
        # Initialize scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Initialize loss
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                features = self.model(batch['texts'])
                loss = train_loss(features, batch['labels'])
                
                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{epochs}: Average Loss = {avg_loss:.4f}")
        
        # Save the model
        self.model.save(output_path)

    # ... (rest of the methods remain the same)

# Example usage with error handling
def main():
    try:
        # Generate example data with smaller size for testing
        print("Generating example data...")
        df1, df2 = generate_example_data(n_rows=100)  # Reduced number of rows for testing
        
        print("\nDataFrame 1 shape:", df1.shape)
        print("DataFrame 1 columns:", df1.columns.tolist())
        print("\nDataFrame 2 shape:", df2.shape)
        print("DataFrame 2 columns:", df2.columns.tolist())
        
        # Initialize trainer
        print("\nInitializing trainer...")
        trainer = ColumnSimilarityTrainer()
        
        # Create training examples
        print("Creating training examples...")
        examples = trainer.create_training_examples([df1, df2])
        print(f"Created {len(examples)} training examples")
        
        if not examples:
            print("Error: No training examples were created!")
            return
        
        # Train the model
        print("\nTraining model...")
        trainer.train(
            examples,
            batch_size=16,
            epochs=3,
            learning_rate=2e-5,
            warmup_steps=50
        )
        
        # Test the model
        print("\nTesting model...")
        test_similarities(trainer, df1, df2)
        
        # Print example of column text representation
        print("\nExample Column Text Representations:")
        print("-" * 50)
        print("\nCustomer ID column:")
        print(trainer.prepare_column_text(df1, 'customer_id'))
        print("\nEmail column:")
        print(trainer.prepare_column_text(df1, 'customer_email'))
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
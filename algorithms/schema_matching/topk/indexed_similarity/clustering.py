import time
import torch
from transformers import AutoTokenizer, AutoModel

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Example texts
texts = ["This is a sentence.", "Another sentence here."] * 100

# Function to get embeddings without batching
def get_embeddings_single(texts):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1))
    return torch.cat(embeddings)

# Function to get embeddings with batching
def get_embeddings_batch(texts, batch_size):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1))
    return torch.cat(embeddings)
print("Start")

# Measure time for single processing
start_time = time.time()
embs = get_embeddings_single(texts)
print(len(texts), len(embs)) 
single_time = time.time() - start_time

# print(f"Time for single processing: {single_time:.2f} seconds")

# Measure time for batch processing
start_time = time.time()
embs = get_embeddings_batch(texts, batch_size=32)
print(len(texts), len(embs))   
batch_time = time.time() - start_time


print(f"Time for batch processing: {batch_time:.2f} seconds")

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Sample data
data = pd.DataFrame({
    'code': ['pN0', 'pNX', 'nan', 'pN2 (FIGO IIIC2)', 'pN1 (FIGO IIIC1)', 'N0', 'N0 (i+)', 'N1', 'N2', 'N3', 'Unknown']
})

# Fill missing values with a placeholder
data['code'].fillna('Unknown', inplace=True)

# Label encoding
label_encoder = LabelEncoder()
data['encoded_code'] = label_encoder.fit_transform(data['code'])

# Custom dataset
class CodeDataset(Dataset):
    def __init__(self, codes):
        self.codes = torch.tensor(codes, dtype=torch.long)

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        return self.codes[idx], self.codes[idx]  # In a real scenario, return (code, target)

# Create dataset and dataloader
dataset = CodeDataset(data['encoded_code'].values)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Define the model
class CodeEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CodeEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1)  # Example: regression problem; adjust as needed

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Model parameters
vocab_size = len(label_encoder.classes_)
embedding_dim = 8

# Initialize the model, loss function, and optimizer
model = CodeEmbeddingModel(vocab_size, embedding_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    for codes, targets in dataloader:
        # Forward pass
        outputs = model(codes)
        loss = criterion(outputs, targets.float())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Extracting embeddings
embeddings = model.embedding.weight.data.numpy()

# Inverse transform to see original codes
decoded_codes = label_encoder.inverse_transform(range(vocab_size))
embedding_dict = {code: embeddings[i] for i, code in enumerate(decoded_codes)}

# Print embedding dictionary
for code, embedding in embedding_dict.items():
    print(f"{code}: {embedding}")

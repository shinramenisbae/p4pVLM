import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

from DataSegmenter import segmentData
from DataLoader import loadAllParticipants
from DataSplitter import splitData


# --- Model Definition ---
class CNN1D_MultiOutput(nn.Module):
    def __init__(self):
        super(CNN1D_MultiOutput, self).__init__()
        self.conv1 = nn.Conv1d(1, 10, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(10)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(10, 20, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(20)
        self.pool2 = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(20 * 35, 600)
        self.dropout = nn.Dropout(0.3)

        # Two output heads: one for valence, one for arousal
        self.fc_valence = nn.Linear(600, 2)
        self.fc_arousal = nn.Linear(600, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, 140, 1) → (batch, 1, 140)
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc_valence(x), self.fc_arousal(x)

# --- Data Loading ---
dataFolder = './data'
allData, allLabels = loadAllParticipants(dataFolder)
trainX, trainY, testX, testY = splitData(allData, allLabels)

# Separate valence and arousal
train_valence = trainY[:, 0]
train_arousal = trainY[:, 1]
test_valence = testY[:, 0]
test_arousal = testY[:, 1]

# Convert to PyTorch tensors
trainX_tensor = torch.tensor(trainX, dtype=torch.float32)
train_valence_tensor = torch.tensor(train_valence, dtype=torch.long)
train_arousal_tensor = torch.tensor(train_arousal, dtype=torch.long)

testX_tensor = torch.tensor(testX, dtype=torch.float32)
test_valence_tensor = torch.tensor(test_valence, dtype=torch.long)
test_arousal_tensor = torch.tensor(test_arousal, dtype=torch.long)

# Dataloaders
batch_size = 128
train_loader = DataLoader(
    TensorDataset(trainX_tensor, train_valence_tensor, train_arousal_tensor),
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(testX_tensor, test_valence_tensor, test_arousal_tensor),
    batch_size=batch_size
)

# --- Training Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN1D_MultiOutput().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# --- Training Loop ---
epochs = 10
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batchX, batchVal, batchAro in train_loader:
        batchX = batchX.to(device)
        batchVal = batchVal.to(device)
        batchAro = batchAro.to(device)

        optimizer.zero_grad()
        outputVal, outputAro = model(batchX)

        lossVal = criterion(outputVal, batchVal)
        lossAro = criterion(outputAro, batchAro)
        loss = lossVal + lossAro

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

# --- Evaluation ---
model.eval()
val_preds, aro_preds = [], []

with torch.no_grad():
    for batchX, _, _ in test_loader:
        batchX = batchX.to(device)
        outVal, outAro = model(batchX)
        val_preds.extend(torch.argmax(outVal, dim=1).cpu().numpy())
        aro_preds.extend(torch.argmax(outAro, dim=1).cpu().numpy())

# --- Reports ---
print("\nTest Report for Valence:")
print(classification_report(test_valence, val_preds))

print("Test Report for Arousal:")
print(classification_report(test_arousal, aro_preds))

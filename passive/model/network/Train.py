
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from CNN import EmotionCNN  # Make sure this outputs only valence logits now
from DataLoader import loadAllParticipants
from imblearn.over_sampling import SMOTE

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

dataFolder = './data'
participant32 = './leaveOneOut'

# Load data
X, y = loadAllParticipants(dataFolder)
X2, y2 = loadAllParticipants(participant32)

#train test split stratified on both, no smote yet
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y[:,0]
)

X_test = np.concatenate((X_test, X2), axis=0)
y_test = np.concatenate((y_test, y2), axis=0)

# convert to tensors for both
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
print(X_train_tensor.shape)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Model, loss, optimiser
model = EmotionCNN()  
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)#criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        # outputs = model(inputs)  # Output shape: (batch_size, 2) for valence logits
        valence_logits, arousal_logits = model(inputs)
        
        valence_targets = targets[:, 0]
        arousal_targets = targets[:, 1]

        valence_loss = criterion(valence_logits, valence_targets)
        arousal_loss = criterion(arousal_logits, arousal_targets)

        #this one for single stream only
        #loss = criterion(outputs, targets)

        loss = valence_loss + arousal_loss


        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)

    # Evaluation after epoch
    model.eval()
    #correct = 0
    valence_correct = 0
    arousal_correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            # #for single stream
            # outputs = model(inputs)
            # _, predicted = torch.max(outputs, 1)
            # correct += (predicted == targets).sum().item()
            # total += targets.size(0)
            valence_logits, arousal_logits = model(inputs)
            valence_preds = torch.argmax(valence_logits, dim=1)
            arousal_preds = torch.argmax(arousal_logits, dim=1)
            
            valence_targets = targets[:, 0]
            arousal_targets = targets[:, 1]
            
            valence_correct += (valence_preds == valence_targets).sum().item()
            arousal_correct += (arousal_preds == arousal_targets).sum().item()
            total += targets.size(0)


    #accuracy = correct / total
    valence_accuracy = valence_correct / total
    arousal_accuracy = arousal_correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Valence Acc: {valence_accuracy * 100:.2f}% - Arousal Acc: {arousal_accuracy * 100:.2f}%")

torch.save(model.state_dict(), "emotion_cnn.pth")
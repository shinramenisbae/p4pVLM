# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class EmotionCNN(nn.Module):
#     def __init__(self):
#         super(EmotionCNN, self).__init__()
        
#         # Same architecture as paper
#         #C1
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm1d(10)
#         #S1
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
#         #C2
#         self.conv2 = nn.Conv1d(in_channels=10, out_channels=20, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm1d(20)
        
#         self.flat_size = 20 * 35
        
#         self.fc1 = nn.Linear(self.flat_size, 700)
#         self.fc2 = nn.Linear(700, 600)
#         self.dropout = nn.Dropout(p=0.3)
        
#         # Separate output heads for valence and arousal
#         self.valence_out = nn.Linear(600, 2)
#         self.arousal_out = nn.Linear(600, 2)

#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, self.flat_size)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.dropout(x)
        
#         valence_out = self.valence_out(x)
#         arousal_out = self.arousal_out(x)
        
#         return valence_out, arousal_out




#valence only
import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 11, kernel_size=5, padding=1)
        self.bn1 = nn.BatchNorm1d(11)
        # self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2) 
        
        self.pool = nn.AvgPool1d(kernel_size=3, stride=3)
        
        self.conv2 = nn.Conv1d(11, 22, kernel_size=5, padding=1)
        self.bn2 = nn.BatchNorm1d(22)
        # self.pool2 = nn.AvgPool1d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv1d(22, 33, kernel_size=5, padding=1)
        self.bn3 = nn.BatchNorm1d(33)
        # self.pool3 = nn.AvgPool1d(kernel_size=3, stride=2)

        
        # Compute flat size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 140)
            # x = self.pool1(F.relu(self.bn1(self.conv1(dummy_input))))
            # x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            # x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            x = self.pool(F.relu(self.bn1(self.conv1(dummy_input))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            self.flat_size = x.view(1, -1).size(1)

        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_size, 700)
        self.fc2 = nn.Linear(700, 600)
        self.fc3 = nn.Linear(600, 400)
        self.dropout = nn.Dropout(p=0.4)

        self.valence_out = nn.Linear(400, 2)
        self.arousal_out = nn.Linear(400, 2)

    def forward(self, x):
        x = x.unsqueeze(1)

        # x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        valence_out = self.valence_out(x)
        arousal_out = self.arousal_out(x)

        return valence_out, arousal_out



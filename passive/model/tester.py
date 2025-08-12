from DataLoader import loadAllParticipants
from DataSplitter import splitData

# Data folder is not in github repo, different directort for now
dataFolder = './data'
allData, allLabels = loadAllParticipants(dataFolder)

print("All data shape:", allData.shape)
print("All labels shape:", allLabels.shape)

# Check the first sample shapes
print("Sample shape (channels, length):", allData[0].shape)
print("First 5 labels:", allLabels[:5])

#Test DataSplitter
trainingData, trainingLabels, validationData, validationLabels, testingData, testingLabels = splitData(allData, allLabels)
print("Training data shape:", trainingData.shape)
print("Training labels shape:", trainingLabels.shape)
print("Validation data shape:", validationData.shape)
print("Validation labels shape:", validationLabels.shape)
print("Testing data shape:", testingData.shape)
print("Testing labels shape:", testingLabels.shape)
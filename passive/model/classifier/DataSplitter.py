# def splitData(allData, allLabels, numParticipants=32, trainParticipants=28):
#     segmentsPerParticipant = len(allData) // numParticipants

#     trainEndIdx = trainParticipants * segmentsPerParticipant

#     trainingData = allData[:trainEndIdx]
#     trainingLabels = allLabels[:trainEndIdx]

#     testingData = allData[trainEndIdx:]
#     testingLabels = allLabels[trainEndIdx:]

#     return trainingData, trainingLabels, testingData, testingLabels
#Around 70% (12,544 segments) for training 15% (2,688) for validation and 15% for testing.
import numpy
from sklearn.model_selection import train_test_split

def splitData(allData, allLabels, trainingSize=0.8, testSize=0.2):
    trainingData, testingData, trainingLabels, testingLabels = train_test_split(allData, allLabels, train_size = trainingSize, random_state = 42)

    return trainingData, trainingLabels, testingData, testingLabels
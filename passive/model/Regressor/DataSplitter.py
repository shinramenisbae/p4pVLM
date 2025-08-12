#Around 70% (12,544 segments) for training 15% (2,688) for validation and 15% for testing.
import numpy
from sklearn.model_selection import train_test_split

def splitData(allData, allLabels, trainingSize=0.7, validationSize=0.15, testSize=0.15):
    trainingData, tempData, trainingLabels, tempLabels = train_test_split(allData, allLabels, train_size = trainingSize, random_state = 42)

    validationData, testingData, validationLabels, testingLabels = train_test_split(tempData, tempLabels, train_size = 0.5, random_state = 42)

    return trainingData, trainingLabels, validationData, validationLabels, testingData, testingLabels
import os
import pickle
import numpy
from DataSegmenter import segmentData

def loadAllParticipants(dataFolder):
    allData = []
    allLabels = []

    for filename in os.listdir(dataFolder):
        if filename.endswith(".dat"):
            filepath = os.path.join(dataFolder, filename)

            with open(filepath, 'rb') as file:
                participant = pickle.load(file, encoding='latin1') #encoding for python 2 (DEAP) and 3 (me) compatibility
            
            #Only use channels 39 to 40 as it is smartwatch related
            data = participant['data'][:, 39:41, :]
            labels = participant['labels']

            for i in range(data.shape[0]):
                trial = data[i]
                label = labels[i]

                segments = segmentData(trial)
                allData.extend(segments)
                allLabels.extend([label] * len(segments))

    return numpy.array(allData), numpy.array(allLabels)

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
            
            #Change to just 39 for ppg since we want 1d
            data = participant['data'][:, 39, :]
            labels = participant['labels']

            for i in range(data.shape[0]):
                trial = data[i]
                label = labels[i]

                segments = segmentData(trial)

                # Only keep valence (index 0) and arousal (index 1), and turn it into a binary classifier
                binaryLabel = [int(label[0] >= 5), int(label[1] >= 5)]

                allData.extend(segments)
                for _ in segments:
                    allLabels.append(binaryLabel)

    return numpy.array(allData), numpy.array(allLabels)

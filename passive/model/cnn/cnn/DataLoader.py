import os
import pickle
import numpy
# from DataSegmenter import segmentData
from DataPreprocess import detrend_ppg, segment_pulses, personal_normalization

def loadAllParticipants(dataFolder):
    allData = []
    allLabels = []

    for filename in os.listdir(dataFolder):
        if filename.endswith(".dat"):
            filepath = os.path.join(dataFolder, filename)

            with open(filepath, 'rb') as file:
                participant = pickle.load(file, encoding='latin1') #encoding for python 2 (DEAP) and 3 (me) compatibility
            
            #Only use channels 39
            data = participant['data'][:, 39, :]
            labels = participant['labels']

            for i in range(data.shape[0]):
                trial = data[i]
                label = labels[i]

                detrended = detrend_ppg(trial)
                person_min = detrended.min()
                person_max = detrended.max()
                if person_max - person_min < 1e-3:
                    continue

                # Segment into pulses
                pulses = segment_pulses(detrended)

                # Normalize pulses
                norm_pulses = personal_normalization(pulses, person_min, person_max)


                # Only keep valence (index 0) and arousal (index 1), and turn it into a binary classifier
                binaryLabel = [int(label[0] >= 5), int(label[1] >= 5)]

                allData.extend(pulses)
                for _ in pulses:
                    allLabels.append(binaryLabel)

    return numpy.array(allData), numpy.array(allLabels)

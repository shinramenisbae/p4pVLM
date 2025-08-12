import numpy
#This may be needed to segment data but idk if it is usable because the emotion label is one for the whole 63 seconds.

#Splits into 8 second chunks where the start point of the segments differ by 4 seconds, we also exclude the first 3 seconds.
#Around 70% (12,544 segments) for training 15% (2,688) for validation and 15% for testing.
def segmentData(data, segmentLength=1024, stepSize=512):
    segments = []
    for start in range(384, data.shape[1], stepSize):
        end = start + segmentLength
        segment = data[:, start:end]

        if segment.shape[1] == segmentLength:
            segments.append(segment)
    
    return segments



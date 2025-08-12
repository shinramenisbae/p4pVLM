import numpy
#This may be needed to segment data but idk if it is usable because the emotion label is one for the whole 63 seconds.

#140 segments over the 63 seconds.
def segmentData(trial, pulseLength=140, stepSize=70):
    segments = []
    for start in range(0, len(trial) - pulseLength + 1, stepSize):
        segment = trial[start:start + pulseLength]
        
        # Normalize each pulse
        segment = (segment - numpy.mean(segment)) / (numpy.std(segment) + 1e-6)
        
        segments.append(segment.reshape(-1, 1))  # Shape (140, 1)
    
    return segments



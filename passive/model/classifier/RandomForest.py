#this is kinda the main class
import numpy
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from DataSplitter import splitData
from DataLoader import loadAllParticipants

dataFolder = './data'
allData, allLabels = loadAllParticipants(dataFolder)
trainingData, trainingLabels, testingData, testingLabels = splitData(allData, allLabels)

#flatten data, random forest needs 1d not 2d
trainingData = trainingData.reshape((trainingData.shape[0], -1))
testingData = testingData.reshape((testingData.shape[0], -1))

# Combine valence and arousal into single multi-class label (0 to 3), each class is a combination of arousal or valence, eg class 0 is 0 arousal 0 valence
trainingLabelsCombined = trainingLabels[:, 0] * 2 + trainingLabels[:, 1]
testingLabelsCombined = testingLabels[:, 0] * 2 + testingLabels[:, 1]

#duplicates minority classes to help balance data using random over sampler
# ros = RandomOverSampler(random_state=42)
# trainingDataResampled, trainingLabelsResampledCombined = ros.fit_resample(trainingData, trainingLabelsCombined)

#same but using Synthetic Minority Over-sampling Technique
smote = SMOTE(random_state=42)
trainingDataResampled, trainingLabelsResampledCombined = smote.fit_resample(trainingData, trainingLabelsCombined)

#training random forest
n_estimators = 300
# rfModel = RandomForestClassifier(n_estimators, random_state=42)
# rfModel.fit(trainingDataResampled, trainingLabelsResampledCombined)

rfModel = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42
)
rfModel.fit(trainingDataResampled, trainingLabelsResampledCombined)

# Predict and decode test labels
testPredsCombined = rfModel.predict(testingData)
valenceTestPreds = testPredsCombined // 2
arousalTestPreds = testPredsCombined % 2

print("Test Report for Valence:")
print(classification_report(testingLabels[:, 0], valenceTestPreds))
print("Test Report for Arousal:")
print(classification_report(testingLabels[:, 1], arousalTestPreds))
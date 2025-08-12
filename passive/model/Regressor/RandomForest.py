#this is kinda the main class
import numpy

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from DataSplitter import splitData
from DataLoader import loadAllParticipants

dataFolder = './data'
allData, allLabels = loadAllParticipants(dataFolder)
trainingData, trainingLabels, validationData, validationLabels, testingData, testingLabels = splitData(allData, allLabels)

trainingData = trainingData.reshape((trainingData.shape[0], -1))
validationData = validationData.reshape((validationData.shape[0], -1))
testingData = testingData.reshape((testingData.shape[0], -1))

#training the random forest
rfModel = RandomForestRegressor(n_estimators=50, random_state=42)

rfModel.fit(trainingData, trainingLabels)


validationPredictions = rfModel.predict(validationData)
#Mean Squared Error (MSE)
mse = mean_squared_error(validationLabels, validationPredictions)
print(f"Mean Squared Error: {mse}")

# R² Score
r2 = r2_score(validationLabels, validationPredictions, multioutput='uniform_average')
print(f"R² Score: {r2}")

#Mean Absolute Error (MAE)
mae = mean_absolute_error(validationLabels, validationPredictions)
print(f"Mean Absolute Error: {mae}")

testingPredictions = rfModel.predict(testingData)

mse_test = mean_squared_error(testingLabels, testingPredictions)
print(f"Test Mean Squared Error: {mse_test}")

r2_test = r2_score(testingLabels, testingPredictions, multioutput='uniform_average')
print(f"Test R² Score: {r2_test}")

mae_test = mean_absolute_error(testingLabels, testingPredictions)
print(f"Test Mean Absolute Error: {mae_test}")
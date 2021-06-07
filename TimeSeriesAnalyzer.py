import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

class TimeSeriesAnalyzer:

    def __init__(self, trainFilePath, testFilePath, columns, window):
        self.window = window
        self.trainData = pd.read_csv(trainFilePath)[columns].values
        self.testData = pd.read_csv(testFilePath)[columns].values
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def transform(self):
        
        # Normalize the training data. 
        self.trainData = self.scaler.fit_transform(self.trainData)

        # Transform the data into input and output variables. 
        self.xTrain, self.yTrain = self.transformTimeSeries(self.trainData)

    def trainLstm(self):
        
        # Fit the training data using a LSTM network. 
        self.model = Sequential()
        self.model.add(LSTM(4, input_shape=(self.window, self.trainData.shape[1])))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

        batchSize = 20
        epochs = 100
        self.model.fit(self.xTrain, self.yTrain, epochs=epochs, batch_size=batchSize)

        #for epoch in range(epochs):
        #    self.model.fit(self.xTrain, self.yTrain, epochs=1, batch_size=batchSize, shuffle=False)
        #    self.model.reset_states()

    def transformTest(self):

        # Transform the test data analogously to the training data. 
        self.testActual = self.testData[:, self.testData.shape[1] - 1]
        self.testData = self.scaler.transform(self.testData)

        # Prepend the last `window` time steps of the training
        # data to the test data. This way, we have enough data
        # to predict all N time steps of the test data. 
        self.testData = np.concatenate((self.trainData[-self.window:, :], self.testData))

        self.xTest, _ = self.transformTimeSeries(self.testData)

    def forecastTimeSeries(self):

        trainPredicted = self.model.predict(self.xTrain)
        testPredicted = self.model.predict(self.xTest)

        trainPredictedExtended = np.zeros((len(trainPredicted), self.trainData.shape[1]))
        trainPredictedExtended[:, self.trainData.shape[1] - 1] = trainPredicted[:, 0]
        trainPredicted = self.scaler.inverse_transform(trainPredictedExtended)[:, self.trainData.shape[1] - 1] 
        testPredictedExtended = np.zeros((len(testPredicted), self.trainData.shape[1]))
        testPredictedExtended[:, self.trainData.shape[1] - 1] = testPredicted[:, 0]
        testPredicted = self.scaler.inverse_transform(testPredictedExtended)[:, self.trainData.shape[1] - 1]   
        self.testPredicted = testPredicted

    # Transforms an array of time series data into structured
    # input and output data that can be used in a supervised
    # learning problem. Assumes the last column contains the 
    # variable to predict. 
    def transformTimeSeries(self, data):
        dataX, dataY = [], []
        for i in range(len(data) - self.window):
            a = data[i:(i + self.window), :]
            dataX.append(a)
            dataY.append(data[i + self.window, data.shape[1] - 1])
        return np.array(dataX), np.array(dataY)

    # For each time step in the testing data, determine whether the 
    # predicted output has moved in the same direction (up or down) 
    # as the actual output, relative to the actual output at the 
    # previous time step. 
    def determineDirectionalAccuracy(self):
        isActualUp = [self.testActual[step] - self.testActual[step - 1] > 0 for step in range(1, self.testActual.shape[0])]
        isPredictedUp = [self.testPredicted[step] - self.testActual[step - 1] > 0 for step in range(1, self.testActual.shape[0])]
        isAccuratelyPredicted = [isUp[0] == isUp[1] for isUp in zip(isActualUp, isPredictedUp)]

        return sum(isAccuratelyPredicted) / len(isAccuratelyPredicted)

    def plot(self):
        plt.plot(self.testActual, 'b-', label='Actual')
        plt.plot(self.testPredicted, 'ro', label='Predicted')
        plt.title('')
        plt.xlabel('Time Steps')
        plt.ylabel('')
        plt.legend()
        plt.show()
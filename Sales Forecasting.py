#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from statsmodels.tools.eval_measures import rmse
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from pandas.tseries.offsets import DateOffset
import warnings
warnings.filterwarnings('ignore')

np.random.seed(15)
def csv_reader(path):
    dataS = pd.read_csv(path)
    return dataS

dataSet = csv_reader('AirPassengers.csv')


dataSet.head(12)



dataSet.Month = pd.to_datetime(dataSet.Month)
dataSet = dataSet.set_index('Month')


dataSet.head(12)



def split_and_scale_dataset(dataS, length):
    x_train, x_test = dataS[:-length], dataS[-length:]
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, scaler



train, test, scaler = split_and_scale_dataset(df, 12)
print("Train shape:")
print(train.shape)
print("Test shape:")
print(test.shape)



nInput = 12
nFeatures = 1
batchSize = 4
nEpochs = 200
generator = TimeseriesGenerator(train, train, length=nInput, batch_size=batchSize)

model = Sequential()
model.add(LSTM(100,activation='relu', input_shape=(nInput, nFeatures)))
model.add(Dropout(0.15))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit_generator(generator, epochs=nEpochs,verbose=1)

model.summary()

predictionList = []

batch = train[-nInput:].reshape((1, nInput, nFeatures))
print(batch)

for i in range(nInput):
    predictionList.append(model.predict(batch)[0])
    batch = np.append(batch[:, 1:, :],[[predictionList[i]]], axis=1)




dataSetPredict = pd.DataFrame(scaler.inverse_transform(predictionList), index=dataSet[-nInput:].index, columns=['Predictions'])

dataSetTest = pd.concat([dataSet, dataSetPredict], axis=1)



dataSetTest.tail(12)



plt.figure(figsize=(20, 5))

plt.plot(dataSetTest.index, dataSetTest['AirPassengers'])
plt.plot(dataSetTest.index, dataSetTest['Predictions'], color='r')
plt.show()



def calculate_rmse(currentVals, predVals, length):
    sum = 0
    for i in range(length):
        sum += math.pow((predVals[i] - currentVals[i]), 2)
    return math.sqrt(sum/length)




print(calculate_rmse(scaler.inverse_transform(train[-nInput:]), scaler.inverse_transform(predictionList), nInput))




train = dataSet


scaler.fit(train)
train = scaler.transform(train)


generator = TimeseriesGenerator(train, train, length=nInput, batch_size=batchSize)

model.fit_generator(generator, epochs=nEpochs, verbose=1)

predictionList = []

batch = train[-nInput:].reshape((1, nInput, nFeatures))

for i in range(nInput):
    predictionList.append(model.predict(batch)[0])
    batch = np.append(batch[:, 1:, :],[[predictionList[i]]], axis=1)



addDates = [dataSet.index[-1] + DateOffset(months=x) for x in range(0, 13)]
futureDates = pd.DataFrame(index=addDates[1:], columns=dataSet.columns)



futureDates.tail(12)



dataSetPredict = pd.DataFrame(scaler.inverse_transform(predictionList),
                         index=futureDates[-nInput:].index, columns=['Predictions'])

dataSetProj = pd.concat([dataSet, dataSetPredict], axis=1)



dataSetProj.tail(12)



plt.figure(figsize=(10, 4))
plt.plot(dataSetProj.index, dataSetProj['AirPassengers'])
plt.plot(dataSetProj.index, dataSetProj['Prediction'], color='r')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

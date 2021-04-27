import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('GOOG.csv', date_parser=True)
# print(data.head())

train_data = data[data['Date'] < '2020-01-01'].copy()
train_data = train_data.drop(['Date', 'Adj Close'], axis=1)

test_data = data[data['Date'] >= '2020-01-01'].copy()
# print(test_data.tail())

sc = MinMaxScaler()
train_data = sc.fit_transform(train_data)
# print(train_data)

x_train = []
y_train = []

# print(train_data.shape[0])

num_day = 60

for i in range(num_day, train_data.shape[0]):
    x_train.append(train_data[i-num_day:i])
    y_train.append(train_data[i, 0])


x_train, y_train = np.array(x_train, dtype=object), np.array(y_train, dtype=object)
# print(x_train.shape, y_train.shape)

from keras import Sequential
import keras
from keras.layers import Dense, Dropout, LSTM

reg = Sequential()

reg.add(LSTM(units = 50, activation='relu', return_sequences=True, input_shape = (x_train.shape[1], 5)))
reg.add(Dropout(0.2))

reg.add(LSTM(units = 60, activation='relu', return_sequences=True))
reg.add(Dropout(0.3))

reg.add(LSTM(units = 80, activation='relu', return_sequences=True))
reg.add(Dropout(0.4))

reg.add(LSTM(units = 120, activation='relu'))
reg.add(Dropout(0.5))

reg.add(Dense(1))

# print(reg.summary())

reg.compile(optimizer=keras.optimizers.Adam(), loss='mean_squared_error')
reg.fit(x_train, y_train, epochs=10, batch_size=32)


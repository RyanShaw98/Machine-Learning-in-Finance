import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler


def feature_label_split(close_values, X_cycle_size, y_cycle_size):
    features = []
    labels = []

    for current_index in range(len(close_values)):
        X_last_index = current_index + X_cycle_size
        y_last_index = X_last_index + y_cycle_size

        if y_last_index > len(close_values):
            break

        X_cycle = close_values[current_index:X_last_index]
        y_cycle = close_values[X_last_index:y_last_index]

        features.append(X_cycle)
        labels.append(y_cycle)

    return np.array(features), np.array(labels)


# --=MAIN=--

currency = "BTC-GBP"  # BTC, ETH, XRP, LTC, BCH

start = dt.datetime(2010, 1, 27)
end_date = dt.datetime(2020, 1, 27)

df = pdr.get_data_yahoo(currency, start, end_date)
df = df[["Close"]]

print(f"df.shape:\n{df.shape}\n")
print(f"df.head:\n{df.head()}\n")

scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
print(f"scaled df.head:\n{df.head()}\n")

# How many days looking back to train
days_to_train = 30
# How many days ahead to predict
days_to_predict = 10
# Features
num_of_features = 1

X, y = feature_label_split(list(df.Close), days_to_train, days_to_predict)

X = X.reshape((X.shape[0], X.shape[1], num_of_features))

model = Sequential()

# Input layer
model.add(LSTM(30, activation="softsign", return_sequences=True, input_shape=(days_to_train, num_of_features)))

# Hidden layer
model.add(LSTM(50, activation='softsign'))

# Output layer
model.add(Dense(days_to_predict))

model.compile(loss="mae", optimizer="adam", metrics=['acc', 'mean_squared_error'])
# Other metrics: 'mean_squared_error', 'mean_squared_logarithmic_error'

model.summary()

epoch_count = 200
batch_size = 32

history = model.fit(X, y, epochs=epoch_count, batch_size=batch_size, validation_split=0.2)

# Code provided by supervisor (Dr Haider Raza) to plot training and validation loss and accuracy

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# End of supervisor code

path = f"deep_learning_models/{currency}_epoch{epoch_count}_batch{batch_size}.h5"
model.save(path)

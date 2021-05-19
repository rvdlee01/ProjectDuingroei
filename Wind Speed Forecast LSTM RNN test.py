import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import calendar
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def readCSV(filename):
    df = pd.read_csv(filename, skipinitialspace=True)
    return df

df = readCSV('WindData.csv')
#x = df[['TG','PG','UG','FG','DDVEC']]
x = df[['FHX']]
#y = df[['FHX']]


#Scale the all of the data to be values between 0 and 1 
scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_data = scaler.fit_transform(x)

train_data = scaled_data

#Splitting the data
x_train=[]
y_train = []

for i in range(10, len(train_data)):
    x_train.append(train_data[i-10:i,0])
    y_train.append(train_data[i,0])
    
#Convert to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data into 3-D array
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# Initialising the RNN
model = Sequential()

model.add(LSTM(units = 100, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))

# Adding a second LSTM layer and Dropout layer
model.add(LSTM(units = 100, return_sequences = True))
model.add(Dropout(0.2))

# Adding a third LSTM layer and Dropout layer
model.add(LSTM(units = 100, return_sequences = True))
model.add(Dropout(0.2))

# Adding a fourth LSTM layer and and Dropout layer
model.add(LSTM(units = 100))
model.add(Dropout(0.2))

# Adding the output layer
# For Full connection layer we use dense
# As the output is 1D so we use unit=1
model.add(Dense(units = 1))

#compile and fit the model on 30 epochs
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, epochs = 20, batch_size = 32, verbose=0)

#check predicted values
predictions = model.predict(x_train) 
#Undo scaling
predictions = scaler.inverse_transform(predictions)

trainingScore = model.evaluate(x_train, y_train, verbose=0)
print('Training Score is : %.2f MSE (%.2f RMSE)' % (trainingScore, math.sqrt(trainingScore)))

plt.plot(df['FHX'], label='actual')
plt.plot(predictions, label='predicted')
plt.show()



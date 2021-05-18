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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#temp
from sklearn.metrics import r2_score

# Get weather data file from https://www.knmi.nl/nederland-nu/klimatologie/daggegevens
def createCSV(filename):
    myCSV = pd.read_csv(filename, skipinitialspace=True)  
    columns = myCSV[['YYYYMMDD','TN','TX','TG','FHN','FHX','FG','RH','PG','UG','DDVEC']] 
    df = columns.copy()
    
    df = df.fillna(0)
    df = df.astype(int)
    df['YYYYMMDD'] = pd.to_datetime(df['YYYYMMDD'], format='%Y%m%d')
    df = df[df['YYYYMMDD'].dt.year > 1995]
    df.to_csv('WindData.csv', index=False)

def readCSV(filename):
    df = pd.read_csv(filename, skipinitialspace=True)
    return df

df = createCSV('weather.csv')
df = readCSV('WindData.csv')
x = df[['TG','PG','UG','FG','DDVEC']] #RH
y = df[['FHX']]

# 90% training data & 10% testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=5, shuffle=False)
#x_train, x_test, y_train, y_test = x[1:8343], x[8343:],x[1:8343], x[8343:]
#0,9 x 9269 = 8343
#0,1 x 9269 = 927

xnorm = StandardScaler()
ynorm = StandardScaler()

x_train=xnorm.fit_transform(x_train)
x_test=xnorm.transform(x_test)
print(x_train)
y_train=ynorm.fit_transform(np.array(y_train).reshape(-1,1))
y_test=ynorm.transform(np.array(y_test).reshape(-1,1))

model = Sequential()
model.add(Dense(256, input_dim=5, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
#print(model.summary())
model.fit(x_train,y_train, epochs=250, batch_size=6000, verbose=0)

# predictions 
trainPredict = model.predict(x_train)
testPredict = model.predict(x_test)

plt.plot(df['FHX'], label='actual data')
plt.plot(range(0,y_train.shape[0]),ynorm.inverse_transform(y_train), label='y_train',alpha=0.6)
plt.plot(range(y_train.shape[0],y_train.shape[0]+y_test.shape[0]),ynorm.inverse_transform(y_test), label='y_test',alpha=0.6)
plt.xlabel('DateTime')
plt.ylabel('Wind Speed')
plt.title('Wind Speed Prediction')
plt.legend()
plt.show()

# Estimate model performance
trainingScore = model.evaluate(x_train, y_train, verbose=0)
print('Accuracy: %.2f' % (trainingScore*100))
print('r2 score for training: ', r2_score(y_train, trainPredict))
print('r2 score for testing: ', r2_score(y_test, testPredict))
print('Training Score is : %.2f MSE (%.2f RMSE)' % (trainingScore, math.sqrt(trainingScore)))
testingScore = model.evaluate(x_test, y_test, verbose=0)
print('Testing Score is  : %.2f MSE (%.2f RMSE)' % (testingScore, math.sqrt(testingScore)))

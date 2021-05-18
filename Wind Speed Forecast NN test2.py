import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import calendar
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split

def createCSV(filename):
    myCSV = pd.read_csv(filename, skipinitialspace=True)  
    columns = myCSV[['YYYYMMDD','TN','TX','TG','FHN','FHX','FG','DDVEC']]
    df = columns.copy()
    
    df = df.fillna(0)
    df = df.astype(int)
    df['YYYYMMDD'] = pd.to_datetime(df['YYYYMMDD'], format='%Y%m%d')
    df = df[df['YYYYMMDD'].dt.year > 1995]
    df.to_csv('WindData.csv', index=False)

def readCSV(filename):
    df = pd.read_csv(filename, skipinitialspace=True)
    return df

#df = createCSV('WeatherHVH.csv')
df = readCSV('WindData.csv')
x = df[['TN','TX','TG','FG','DDVEC']]
y = df[['FHX']]

# 90% training data & 10% testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=4)

xnorm = StandardScaler()
ynorm = StandardScaler()

x_train=xnorm.fit_transform(x_train)
x_test=xnorm.transform(x_test)
y_train=ynorm.fit_transform(np.array(y_train).reshape(-1,1))
y_test=ynorm.transform(np.array(y_test).reshape(-1,1))

model = Sequential()
model.add(Dense(32, input_dim=5, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
#print(model.summary())

model.fit(x_train,y_train, epochs=200, batch_size=1000, verbose=0)

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
print('Training Score is : %.2f MSE (%.2f RMSE)' % (trainingScore, math.sqrt(trainingScore)))
testingScore = model.evaluate(x_test, y_test, verbose=0)
print('Testing Score is  : %.2f MSE (%.2f RMSE)' % (testingScore, math.sqrt(testingScore)))

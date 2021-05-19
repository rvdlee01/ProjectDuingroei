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
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#Get weather data file from https://www.knmi.nl/nederland-nu/klimatologie/daggegevens
#TN = Minimale temperatuur in 0,1 graden celcius
#TX = Maximale temperatuur in 0,1 graden celcius
#TG = Etmaalgemiddelde temperatuur in 0,1 graden celcius
#FHN = Laagste uurgemiddelde windsnelheid in 0.1 m/s
#FHX = Hoogste uurgemiddelde windsnelheid in 0.1 m/s
#FG = Etmaalgemiddelde uurgemiddelde windsnelheid in 0.1 m/s
#RH = Etmaalsom van neerslag in 0.1 mm (-1 = kleiner dan 0.5 mm)
#PG = Etmaalgemiddelde luchtdruk herleidt tot zeeniveau in 0.1 hpa. Berekend uit 24 uur waarden
#UG = Etmaalgemiddelde relatieve vochtigheid in %
#DDVEC = Vectorgemiddelde windrichting in graden (360 graden is noorden etc., 0 = windstil)
def createCSV(filename):
    myCSV = pd.read_csv(filename, skipinitialspace=True)  
    df = myCSV[['YYYYMMDD','TN','TX','TG','FHN','FHX','FG','RH','PG','UG','DDVEC']]
    
    df = df.fillna(0)
    df = df.astype(int)
    df['YYYYMMDD'] = pd.to_datetime(df['YYYYMMDD'], format='%Y%m%d')
    df = df[df['YYYYMMDD'].dt.year > 1995]
    df.to_csv('WindData.csv', index=False)

def readCSV(filename):
    df = pd.read_csv(filename, skipinitialspace=True)
    return df

#df = createCSV('weather.csv') # Uncomment if you do not have the wind data csv file
df = readCSV('WindData.csv')
x = df[['TG', 'TN', 'TX','PG','UG', 'FHN', 'FG', 'DDVEC', 'RH']]
y = df[['FHX']]
df['YYYYMMDD'] = pd.to_datetime(df['YYYYMMDD'])
#90% training data & 10% testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0, shuffle=False)

#Standardize data
xnorm = StandardScaler()
ynorm = StandardScaler()

x_train = xnorm.fit_transform(x_train)
x_test = xnorm.transform(x_test)

y_train = ynorm.fit_transform(np.array(y_train).reshape(-1,1))
y_test = ynorm.transform(np.array(y_test).reshape(-1,1))

#Adding dense layers
model = Sequential()
model.add(Dense(256, input_dim=9, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

#print(model.summary())
model.fit(x_train,y_train, epochs=300, batch_size=6000, verbose=0, shuffle=False)

#predictions 
trainPredict = model.predict(x_train)
testPredict = model.predict(x_test)

def plotPrediction(actualDataframe, trainPredict, testPredict, ylabelname, title):
    plt.plot(df['YYYYMMDD'],actualDataframe, label='actual data')
    plt.plot(df['YYYYMMDD'][:y_train.shape[0]],ynorm.inverse_transform(trainPredict), label='training prediction',alpha=0.6)
    plt.plot(df['YYYYMMDD'][y_train.shape[0]:y_train.shape[0]+y_test.shape[0]],ynorm.inverse_transform(testPredict), label='testing prediction',alpha=0.6)
    plt.xlabel('DateTime')
    plt.ylabel(ylabelname)
    plt.title(title)
    plt.legend()
    plt.show()

plotPrediction(df['FHX'], trainPredict, testPredict, 'Wind Speed', 'Wind Speed Prediction')

def dataframeToCSV(actualDataframe, filename):
    #convert np arrays to list and add the training and testing results together in a list
    listOfTraining = trainPredict.tolist()
    listOfTesting = testPredict.tolist()
    listOfActual = actualDataframe.tolist()
    joinedList = listOfTraining + listOfTesting

    #convert array(list[]) to integer
    predictedList = []
    count = 0
    for date in df['YYYYMMDD']:
        predictedList.append(int(float(ynorm.inverse_transform(joinedList[count]))))
        count = count + 1

    #assign list to column
    newDataframe = {'Date': df['YYYYMMDD'].tolist(), 'Predicted': predictedList, 'Actual': listOfActual}

    #create dataframe
    OutputDataframe = pd.DataFrame(newDataframe)

    #convert dataframe to csv
    OutputDataframe.to_csv(filename, index=False)

dataframeToCSV(df['FHX'], 'WeatherPredictionOutput.csv')

def showPredictionYear(filename, year):
    new_df = pd.read_csv(filename, skipinitialspace=True)
    new_df['Date'] = pd.to_datetime(new_df['Date'])
    
    new_df = new_df[new_df['Date'].dt.year == year]
    
    plt.plot(new_df['Date'],new_df['Actual'],label='actual')
    plt.plot(new_df['Date'],new_df['Predicted'],label='predicted')
    plt.show()

showPredictionYear('WeatherPredictionOutput.csv', 2021)

#Estimate model performance
def evaluationResults(x, y, predict):
    score = model.evaluate(x, y, verbose=0)
    print('Training Score is : %.2f MSE (%.2f RMSE)' % (score, math.sqrt(score)))
    print('r2 score for training: ', r2_score(y, predict))
    print('')

evaluationResults(x_train, y_train, trainPredict)
evaluationResults(x_test, y_test, testPredict)

#get max error rate
count = 0
maxerrorrate = 0
for y in testPredict:
    if((ynorm.inverse_transform(testPredict)[count] - ynorm.inverse_transform(y_test)[count])/ynorm.inverse_transform(y_test)[count] * 100 > maxerrorrate):
        maxerrorrate = (ynorm.inverse_transform(testPredict)[count] - ynorm.inverse_transform(y_test)[count])/ynorm.inverse_transform(y_test)[count] * 100
    count = count + 1
print('Highest error rate: ', maxerrorrate)

#new instances where we do not know the answer

#make a prediction

#show the predicted outputs

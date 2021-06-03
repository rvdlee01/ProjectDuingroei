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

def readCSV(filename):
    df = pd.read_csv(filename, skipinitialspace=True)
    return df

df = readCSV('newdataColumns.csv')
df=df[df['year'] > 2002]

df['YYYYMMDD'] = df['year']
x = df[['windkracht6','windkracht7','windkracht8','windkracht9','windkracht10','windkracht11','windkracht12','north','east','south','west','northeast','southeast','southwest','northwest','highhumidity','lowhumidity','aveghumidity','neerslag']]
y = df['punt3'].values
#90% training data & 10% testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

#Standardize data
xnorm = StandardScaler()
ynorm = StandardScaler()

x_train = xnorm.fit_transform(x_train)
x_test = xnorm.transform(x_test)

y_train = ynorm.fit_transform(np.array(y_train).reshape(-1,1))
y_test = ynorm.transform(np.array(y_test).reshape(-1,1))

#Adding dense layers
model = Sequential()
model.add(Dense(128, input_dim=19, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

#print(model.summary())
model.fit(x_train,y_train, epochs=10, batch_size=50, verbose=0)

#predictions 
trainPredict = model.predict(x_train)
testPredict = model.predict(x_test)

def plotPrediction(actualDataframe, trainPredict, testPredict, ylabelname, title):
    #plt.plot(df['YYYYMMDD'],actualDataframe, label='actual data')
    plt.plot(df['YYYYMMDD'][:y_train.shape[0]],ynorm.inverse_transform(y_train), label='training actual values',alpha=0.6)
    plt.plot(df['YYYYMMDD'][:y_train.shape[0]],ynorm.inverse_transform(trainPredict), label='training prediction',alpha=0.6)
    plt.plot(df['YYYYMMDD'][y_train.shape[0]:y_train.shape[0]+y_test.shape[0]],ynorm.inverse_transform(y_test), label='testing actual values',alpha=0.6)
    plt.plot(df['YYYYMMDD'][y_train.shape[0]:y_train.shape[0]+y_test.shape[0]],ynorm.inverse_transform(testPredict), label='testing prediction',alpha=0.6)
    plt.xlabel('DateTime')
    plt.ylabel(ylabelname)
    plt.title(title)
    plt.legend()
    plt.show()

plotPrediction(df['punt1'], trainPredict, testPredict, 'Wind Speed', 'Wind Speed Prediction')

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

dataframeToCSV(df['punt1'], 'TESTTESTTEST.csv')

def showPredictionYear(filename, year):
    new_df = pd.read_csv(filename, skipinitialspace=True)
    new_df['Date'] = pd.to_datetime(new_df['Date'])
    
    new_df = new_df[new_df['Date'].dt.year == year]
    
    plt.plot(new_df['Date'],new_df['Actual'],label='actual')
    plt.plot(new_df['Date'],new_df['Predicted'],label='predicted')
    plt.show()

#showPredictionYear('TESTTESTTEST.csv', 2020)

#Estimate model performance
def evaluationResults(x, y, predict):
    score = model.evaluate(x, y, verbose=0)
    print('Training Score: %.2f MSE (%.2f RMSE)' % (score, math.sqrt(score)))
    print('r2 score: ', r2_score(y, predict))
    print('')

evaluationResults(x_train, y_train, trainPredict)
evaluationResults(x_test, y_test, testPredict)

#get max error rate
count = 0
maxerrorrate = 0
minerrorrate = 0
for y in testPredict:
    if((ynorm.inverse_transform(testPredict)[count] - ynorm.inverse_transform(y_test)[count])/ynorm.inverse_transform(y_test)[count] * 100 > maxerrorrate):
        maxerrorrate = (ynorm.inverse_transform(testPredict)[count] - ynorm.inverse_transform(y_test)[count])/ynorm.inverse_transform(y_test)[count] * 100
    elif((ynorm.inverse_transform(testPredict)[count] - ynorm.inverse_transform(y_test)[count])/ynorm.inverse_transform(y_test)[count] * 100 < minerrorrate):
        minerrorrate = (ynorm.inverse_transform(testPredict)[count] - ynorm.inverse_transform(y_test)[count])/ynorm.inverse_transform(y_test)[count] * 100
    count = count + 1
print('Highest error rate (max): ', maxerrorrate)
print('Highest error rate (min): ', minerrorrate)

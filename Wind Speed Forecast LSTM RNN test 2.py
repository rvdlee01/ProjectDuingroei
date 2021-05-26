import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# create future forecast dates
def create_dates(start,days):
    v = pd.date_range(start=start, periods=days+1, freq='D', closed='right')
    list_of_dates = pd.DataFrame(index=v)
    return list_of_dates

# train-test split for a user input ratio
def train_test_split(df, column_name, ratio):
    nrow = len(df)
    print(column_name+' total samples: ',nrow)
    split_row = int((nrow)*ratio)
    print('Training samples: ',split_row)
    print('Testing samples: ',nrow-split_row)
    # Split rows
    train = df.iloc[:split_row]
    test = df.iloc[split_row:]
    return train, test, split_row     

# data transformation
def data_transformation(x_train,x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)

    # Creates new dataframes for scaled data
    x_train_scaled_df = pd.DataFrame(x_train_scaled, index = x_train.index, columns=[x_train.columns[0]])
    x_test_scaled_df = pd.DataFrame(x_test_scaled, index = x_test.index, columns=[x_test.columns[0]])

    return x_train_scaled_df, x_test_scaled_df, scaler  

# feature builder - This section creates feature set with lag number of predictors--Creating features using lagged data
def timeseries_feature_builder(df, lag):
    df_copy = df.copy()
    for i in range(1,lag):
        df_copy['lag'+str(i)] = df.shift(i)
    return df_copy

# preprocessing -- drop null values and make arrays 
def make_arrays(x_train,x_test):
    x_train_array = x_train.dropna().drop(x_train.columns[0], axis=1).values
    y_train_array = x_train.dropna()[x_train.columns[0]].values
    x_test_array = x_test.dropna().drop(x_test.columns[0], axis=1).values
    y_test_array = x_test.dropna()[x_test.columns[0]].values
    return x_train_array, y_train_array, x_test_array, y_test_array

# Define LSTM model
def lstm_model(units, trainX, testX, y_train_array, y_test_array):
    model = Sequential()
    model.add(LSTM(units,return_sequences=True, input_shape=(trainX.shape[1],trainX.shape[2])))
    model.add(Dropout(0.2))    
    model.add(LSTM(units, return_sequences=True))
    model.add(Dropout(0.2))   
    model.add(LSTM(units))
    model.add(Dropout(0.2))
    model.add(Dense(1))        
    model.compile(optimizer='adam', loss='mean_squared_error')
    #300, 150
    model.fit(trainX, y_train_array, batch_size=300, epochs=150, validation_data=(testX, y_test_array), verbose=0)
    return model

# validation result 
def valid_result(model, testX, y_test_array, scaler, all_windspeed, split_row, lag):    
    testPredict = model.predict(testX)
    rSquare_test = r2_score(y_test_array, testPredict)
    print('Test R-squared is: %f'%rSquare_test)    
    testPredict = scaler.inverse_transform(testPredict)        
    new_test = all_windspeed.iloc[split_row:]       
    x_test_pred = new_test.iloc[lag:].copy()
    x_test_pred['Forecast'] = testPredict
    return x_test_pred 

# multi step future forecast for next days number of days. 
def forecast(model, testX, test_tract1, lag, scaler, days):
    one_year = []
    new0 = testX[-1]        
    last = test_tract1.iloc[-1]
    new_predict = last[0]        
    new_array = np.insert(new0, 0, new_predict)        
    new_array = np.delete(new_array, -1)
    new_array_reshape = np.reshape(new_array, (-1,1,lag))       
    new_predict = model.predict(new_array_reshape)
    temp_predict = scaler.inverse_transform(new_predict) 
    one_year.append(temp_predict[0][0].round(2))
    
    for i in range(1,days):
        new_array = np.insert(new_array, 0, new_predict)             
        new_array = np.delete(new_array, -1)
        new_array_reshape = np.reshape(new_array, (-1,1,lag))            
        new_predict = model.predict(new_array_reshape)
        temp_predict = scaler.inverse_transform(new_predict) 
        one_year.append(temp_predict[0][0].round(2))
    return one_year     

def windspeed_lstm(all_windspeed, lag, days):    

    # Create list of dates
    one_year_forecast_lstm = create_dates('2021-05-17',days)

    windspeed_df = all_windspeed
    column_name = 'FHX'

    # Split data into 80% training data and 20% test data
    x_train, x_test, split_row = train_test_split(windspeed_df, column_name, 0.80)

    # Scale data
    x_train_scaled_df, x_test_scaled_df, scaler = data_transformation(x_train,x_test)

    x_train = timeseries_feature_builder(x_train_scaled_df, lag+1) 
    x_test = timeseries_feature_builder(x_test_scaled_df, lag+1)
    print(x_train)

    # Create arrays
    x_train_arr, y_train_arr, x_test_arr, y_test_arr = make_arrays(x_train,x_test)

    trainX = np.reshape(x_train_arr, (x_train_arr.shape[0],1,x_train_arr.shape[1])) 
    testX = np.reshape(x_test_arr, (x_test_arr.shape[0],1,x_test_arr.shape[1]))

    # LSTM modelling & forecast
    model = lstm_model(30, trainX, testX, y_train_arr, y_test_arr)

    x_test_pred = valid_result(model, testX, y_test_arr, scaler, windspeed_df, split_row, lag)
    one_year = forecast(model, testX, x_test, lag, scaler, days)
    one_year_forecast_lstm[column_name] = np.array(one_year)

    print(one_year_forecast_lstm[column_name])
    # Plot result
    plt.plot(x_test_pred)        
    plt.plot(one_year_forecast_lstm[column_name], color='red', label='forecast')         
    plt.ylabel('Windspeed in 0.1 m/s')
    plt.legend(loc='upper right')
    plt.show()        

    return one_year_forecast_lstm

def readCSV(filename):
    df = pd.read_csv(filename, skipinitialspace=True)
    return df

def createCSV(filename):
    myCSV = pd.read_csv(filename, skipinitialspace=True)  
    df = myCSV[['YYYYMMDD','FHX']]
    
    df = df.fillna(0)
    df = df.astype(int)
    df['YYYYMMDD'] = pd.to_datetime(df['YYYYMMDD'], format='%Y%m%d')
    df = df[df['YYYYMMDD'].dt.year > 1995]
    df.to_csv('ActualData.csv', index=False)

#newCSV = createCSV('etmgeg_330.csv')

df = readCSV('WindData.csv')
x = df[['YYYYMMDD','FHX']]

new_df = x.copy()
new_df['YYYYMMDD'] = pd.to_datetime(new_df['YYYYMMDD'])
# set date column as index of the dataset
new_df = new_df.set_index('YYYYMMDD')
new_df = new_df.sort_index()
all_windspeed = new_df
print(all_windspeed.tail())

# Fitting and forecast using LSTM  -- output of train loss and valid loss is turned off
lstm_prediction = windspeed_lstm(all_windspeed,730,365)



















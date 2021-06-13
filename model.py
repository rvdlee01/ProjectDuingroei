import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('newdataColumns.csv')
df=df[df['jaar'] > 2002]

df['YYYYMMDD'] = df['jaar']
x = df[['windkracht6','windkracht7','windkracht8','windkracht9','windkracht10','windkracht11','windkracht12','north','east','south','west','northeast','southeast','southwest','northwest','highhumidity','lowhumidity','aveghumidity','neerslag']]
y = df['duinhoogte'].values
    
#90% training data & 10% testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0, shuffle=False)
   
#Standardize data
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

y_train = scaler.fit_transform(np.array(y_train).reshape(-1,1))
y_test = scaler.transform(np.array(y_test).reshape(-1,1))

#Adding dense layers
model = Sequential()
#model.add(Dense(64, input_dim=19, activation='relu'))
model.add(Dense(4, input_dim=19, activation='relu'))
#model.add(Dense(8, activation='linear'))
#model.add(Dense(2, activation='linear')) #maybe sigmoid? max 3 layers!
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='rmsprop')

#train model
model.fit(x_train,y_train, epochs=100, batch_size=10, verbose=0, shuffle=False)

# save the model to disk
model.save('model')

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
import seaborn as sns
import csv

filename = "test.csv"

df = pd.read_csv(filename)
X = df[['jaar', 'windkracht7', 'windkracht8', 'windkracht9', 'windkracht10', 'windkracht11', 'neerslag']]
X2 = df[['jaar', 'stormdagen']]
y = df['duinhoogte']
regr = linear_model.LinearRegression()

#predict functions_________________________________________________________
def predictDuneheight(year, w7, w8, w9, w10, w11, precipitation):
    regr.fit(X,y)
    return regr.predict([[year, w7, w8, w9, w10, w11, precipitation]])

def predictDuneheightWithStormdays(year, stormdays):
    regr.fit(X2,y)
    return regr.predict([[year, stormdays]])

#test with only duneheights and stormdata_____________________________________

print("Duinhoogtes bij jaren met 46 stormdagen: ")
print("Duinhoogte 2019: ") #actual height is 851 cm
print(predictDuneheightWithStormdays(2019, 46)) #851 cm
print("Duinhoogte 2020: ") #actual height is 941 cm
print(predictDuneheightWithStormdays(2020, 46))#842 cm
print("Duinhoogte 2021: ") #predicted with 46 stormdagen
print(predictDuneheightWithStormdays(2021, 46))#833 cm

#test with only duneheights and windpower_____________________________________

                #actual height is 969 cm
print("2008: actual height is 969 cm")
print("Duinhoogte 2008 met gedetailleerde gegevens:")
print(predictDuneheight(2008,35,14,3,0,0,869.8)) #963 cm

print("Duinhoogte 2008 met alleen stormdagen:")
print(predictDuneheightWithStormdays(2008,52)) #960 cm

                #actual height is 944 cm
print("2003: actual height is 944 cm")
print("Duinhoogte 2003 met gedetailleerde gegevens:")
print(predictDuneheight(2003,24,6,0,0,0,637.5)) #934 cm

print("Duinhoogte 2003 met alleen stormdagen:")
print(predictDuneheightWithStormdays(2003,30)) #974 cm

#print all predictions________________________________________________________

with open(filename, newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    del data[0] #remove list with column names

duneheights = []
realduneheights = []
years = []
count = 0

#list predicted duneheights and actual duneheights_____________________________
for line in data:
  realduneheights.append(float(line[1]))
  del line[1:3] #remove stormdagen & duinhoogtes from list
  
  for i in range(0, len(line)):
    line[i] = int(float(line[i]))#convert values to floats
    
  line[0] = int(line[0]) #convert year back to integer
  duneheights.append(float(predictDuneheight(line[0], line[1], line[2],line[3],line[4],line[5],line[6])))
  years.append(count)
  count += 1

#test plotting graph____________________________________________________________

x=years
y=duneheights
y2=realduneheights

plt.plot(x,y, label="predicted duneheight")
plt.plot(x,y2, label="actual duneheight")

plt.xlabel('year')
plt.ylabel('duneheights')
plt.title('Predicted duneheight vs actual duneheight 2003 - 2019')
plt.legend()

sns.set_theme(color_codes=True)

plt.show()

#Print actual duneheights and predicted duneheights____________________________

print("YEAR - PREDICTED DUNEHEIGHT - ACTUAL DUNEHEIGHT")
year = 2003
for i in range(0, len(duneheights)):
    print(year, " - ", duneheights[i], " - ", realduneheights[i])
    year += 1



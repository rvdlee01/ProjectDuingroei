import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
import seaborn as sns
import csv
from numpy import median
from sklearn.metrics import r2_score

filename = "dunedata.csv"

df = pd.read_csv(filename)
X = df[['jaar','windkracht7', 'windkracht8', 'windkracht9', 'windkracht10', 'windkracht11', 'windkracht12', 'neerslag']]
y = df['punt1']
regr = linear_model.LinearRegression()
regr.fit(X,y)

#predict functions_________________________________________________________
def predictDuneheight(year, w7, w8, w9, w10, w11, w12, precipitation):
    return regr.predict([[year, w7, w8, w9, w10, w11, w12, precipitation]])

#add csv data to "data" variable____________________________________________

with open(filename, newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    del data[0] #remove list with column names

duneheights = []
realduneheights = []
years = []
count = 0

#list actual duneheights______________________________________________________
print("Preview of dataset")
for line in data:
  realduneheights.append(float(line[1]))
  del line[1:5] #remove stormdagen & dune heights from list
  
  for i in range(0, len(line)):
    line[i] = int(float(line[i]))#convert values to floats
    
  print(line)
  duneheights.append(float(predictDuneheight(line[0], line[1], line[2],line[3],line[4],line[5],line[6],line[7])))
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
plt.title('Predicted duneheight vs actual duneheight 2003 - 2020')
plt.legend()

sns.set_theme(color_codes=True)

plt.show()

#Print actual duneheights and predicted duneheights____________________________
print('')
print("YEAR - PREDICTED DUNEHEIGHT - ACTUAL DUNEHEIGHT - ERROR RATE")
year = 2003
for i in range(0, len(duneheights)):
    print(year, " - ", duneheights[i], " - ", realduneheights[i])
    year += 1

#calculate outliers in error rate
diffOutput = []
print('')
print('Error rate:')
count = 0
for height in realduneheights:
    diffOutput.append((duneheights[count] - realduneheights[count])/realduneheights[count]*100)
    print(count, " : ", diffOutput[count])
    count = count + 1
    
diffOutput.sort()

arr = diffOutput
q1 = np.quantile(arr, .25)
q3 = np.quantile(arr, .75)
median = np.quantile(arr, .50)
interkwartielafstand = q3 - q1
binnengrensQ1 = q1 - (interkwartielafstand * 1.5)
binnengrensQ3 = q3 + (interkwartielafstand * 1.5)
buitengrensQ1 = q1 - (interkwartielafstand * 4.5)
buitengrensQ3 = q3 + (interkwartielafstand * 4.5)

print('')
print("Median of arr : ", median)
print("Q1 quantile of arr : ", q1)
print("Q3 quantile of arr : ", q3)
print("interkwartielafstand : ", interkwartielafstand)
print("binnengrenzen zijn : ", binnengrensQ1, " en ",
                               binnengrensQ3)
print("buitengrenzen zijn : ", buitengrensQ1, " en ",
                               buitengrensQ3)

outliers = []
extremeOutliers = []

print('')
print("Uitschieters:")
for diff in diffOutput:
    #(NOT here)----OutsideQ1----(HERE)----WithinQ1----(NOT here)----WithinQ3----(HERE)----OutsideQ3----(NOT here)
    if((buitengrensQ1 <= diff and diff < binnengrensQ1) or
       binnengrensQ3 < diff and diff <= buitengrensQ3):
        outliers.append(diff)
        print(diff)
    #(HERE)----OutsideQ1----(NOT here)----OutsideQ3----(HERE)
    elif(diff < buitengrensQ1 or buitengrensQ3 < diff):
        extremeOutliers.append(diff)
        
print('')        
print("r2 score", r2_score(realduneheights, duneheights))

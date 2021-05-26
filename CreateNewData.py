import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import calendar

def createCSV(filename):
    myCSV = pd.read_csv(filename, skipinitialspace=True)  
    columns = myCSV[['YYYYMMDD','FHX', 'RH', 'DDVEC']]
    df = columns.copy()
    
    df = df.fillna(0)
    df = df.astype(int)
    df['YYYYMMDD'] = pd.to_datetime(df['YYYYMMDD'], format='%Y%m%d')
    df = df[df['YYYYMMDD'].dt.year > 1995]
    df.to_csv('newdata.csv', index=False)

def readCSV(filename):
    df = pd.read_csv(filename, skipinitialspace=True)
    return df

newCSV = createCSV('etmgeg_330.csv')


# Create new CSV

df = readCSV('newdata.csv')
df['YYYYMMDD'] = pd.to_datetime(df['YYYYMMDD'])

df.FHX = (df.FHX.values*0.1).round(1)
df.RH = (df.RH.values*0.1).round(1)
# Filter dataframe on windpower above 6
df = df[df.FHX.values >= 6.0]
# Filter dataframe on precipitation smaller than 10mm
df = df[df.RH.values < 10.0]


dataList = []
yearList = []
for i in df['YYYYMMDD'].dt.year:
    if (i not in yearList):
        yearList.append(i)

for y in yearList:
    # Retrieves data from specific year
    yeardata = df[df['YYYYMMDD'].dt.year == y]

    wp6,wp7,wp8,wp9,wp10,wp11,wp12 = 0,0,0,0,0,0,0
    # Windschaal van Beaufort
    for value in yeardata.FHX.values:
        if((value >= 10.8 and value <= 13.8)):
            wp6 += 1
        if(value >= 13.9 and value <= 17.1):
            wp7 += 1
        if(value >= 17.2 and value <= 20.7):
            wp8 += 1
        if(value >= 20.8 and value <= 24.4):
            wp9 += 1
        if(value >= 24.5 and value <= 28.4):
            wp10 += 1
        if(value >= 28.5 and value <= 32.6):
            wp11 += 1
        if(value > 32.6):
            wp12 += 1
        # year, windpower => 6, 7, 8, 9, 10, 11, 12
        data = [y,wp6,wp7,wp8,wp9,wp10,wp11,wp12]
    dataList.append(data)
newdf = pd.DataFrame(dataList, columns = ['year', 'windkracht6','windkracht7','windkracht8','windkracht9','windkracht10','windkracht11','windkracht12'])
newdf.to_csv('newdataColumns.csv', index=False)


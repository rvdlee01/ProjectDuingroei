import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import calendar

def createCSV(filename):
    myCSV = pd.read_csv(filename, skipinitialspace=True)  
    columns = myCSV[['YYYYMMDD','FHX', 'RH', 'DDVEC', 'UG']]
    #DDVEC : 360=noord, 90=oost, 180=zuid, 270=west, 0=windstil
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
df.UG = (df.UG.values/100).round(1)
# Filter dataframe on windpower above 6
df = df[df.FHX.values >= 10.8]
# Filter dataframe on precipitation smaller than 10mm
df = df[df.RH.values < 10.0]

dataList = []
yearList = []
for i in df['YYYYMMDD'].dt.year:
    if (i not in yearList):
        yearList.append(i)

def addWinddirection(degree,north,east,south,west,northeast,southeast,southwest,northwest):
    if((degree > 337 and degree <= 360) or (degree > 0 and degree <= 22)): #north
        north += 1
    elif(degree > 67 and degree <= 112): #east
        east += 1
    elif(degree > 157 and degree <= 202): #south
        south += 1
    elif(degree > 247 and degree <= 292): #west
        west += 1
    elif(degree > 22 and degree <= 67): #northeast
        northeast += 1
    elif(degree > 112 and degree <= 157): #southeast
        southeast += 1
    elif(degree > 202 and degree <= 247): #southwest
        southwest += 1
    elif(degree > 292 and degree <= 337): #northwest
        northwest += 1
    return north,east,south,west,northeast,southeast,southwest,northwest

for y in yearList:
    # Retrieves data from specific year
    yeardata = df[df['YYYYMMDD'].dt.year == y]

    wp6,wp7,wp8,wp9,wp10,wp11,wp12,north,east,south,west,northeast,southeast,southwest,northwest,highhumidity, lowhumidity, aveghumidity = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    # Windschaal van Beaufort
    for value, degree, humidity in yeardata[['FHX', 'DDVEC', 'UG']].values:
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
        #increase winddirection
        north,east,south,west,northeast,southeast,southwest,northwest = addWinddirection(degree,north,east,south,west,northeast,southeast,southwest,northwest)
        if(humidity < 0.5):
            lowhumidity += 1
        elif(humidity >= 0.5 and humidity <= 0.6):
            aveghumidity += 1
        else:
            highhumidity += 1
    data = [y,wp6,wp7,wp8,wp9,wp10,wp11,wp12,north,east,south,west,northeast,southeast,southwest,northwest,highhumidity, lowhumidity, aveghumidity]
    print(data)
    dataList.append(data)
newdf = pd.DataFrame(dataList, columns = ['year', 'windkracht6','windkracht7','windkracht8','windkracht9','windkracht10','windkracht11','windkracht12',
                                          'north','east','south','west','northeast','southeast','southwest','northwest','highhumidity', 'lowhumidity', 'aveghumidity'])
newdf.to_csv('newdataColumns.csv', index=False)


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
    df = df[df['YYYYMMDD'].dt.year > 1996]
    df = df[df['YYYYMMDD'].dt.year <= 2020]
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

punt1 = [928, 964, 913, -1, 904, -1, 944, 930, 1007, 992, 1006, 969, 936, 873, 947, 1003, 953, 930, 901, 877,
         855, 849, 851, 841]

punt2 = [791, 826, 818, -1, 806, -1, 850, 836, 861, 858, 887, 961, 831, 827, 946, 949, 917, 888, 850, 830, 822,
         808, 803, 800]

punt3 = [814, 848, 831, -1, 807, -1, 877, 867, 891, 920, 922, 995, 897, 849, 1067, 1065, 1060, 1034, 1013, 985,
         978, 951, 937, 924]

neerslag = [656.5, 1206.7, 1020.6, 971.4, 1154.7, 974.5, 637.5, 840.8, 794.7, 883.5, 892.6, 869.8, 746.2, 850.8,
            897.9, 994, 886.4, 804.4, 891.3, 865.8, 956.3, 700.9, 929.3, 985.3]

count = 0
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
    data = [y,punt1[count],punt2[count],punt3[count],wp6,wp7,wp8,wp9,wp10,wp11,wp12,north,east,south,west,northeast,southeast,southwest,northwest,highhumidity, lowhumidity, aveghumidity,neerslag[count]]
    print(data)
    dataList.append(data)
    count += 1
newdf = pd.DataFrame(dataList, columns = ['year','punt1','punt2','punt3','windkracht6','windkracht7','windkracht8','windkracht9','windkracht10','windkracht11','windkracht12',
                                          'north','east','south','west','northeast','southeast','southwest','northwest','highhumidity', 'lowhumidity', 'aveghumidity','neerslag'])
newdf.to_csv('newdataColumns.csv', index=False)


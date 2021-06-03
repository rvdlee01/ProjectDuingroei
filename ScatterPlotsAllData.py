import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import calendar

myCSV = pd.read_csv('newdataColumns.csv',skipinitialspace=True)

# Dune heights
duneheight1 = myCSV.punt1
duneheight2 = myCSV.punt2
duneheight3 = myCSV.punt3

# Wind direction
north, east, south, west = myCSV.north, myCSV.east, myCSV.south, myCSV.west
northeast, southeast, southwest, northwest = myCSV.northeast, myCSV.southeast, myCSV.southwest, myCSV.northwest

# Humidity
highhumidity, lowhumidity, avghumidity = myCSV.highhumidity, myCSV.lowhumidity, myCSV.aveghumidity

# Precipitation
precipitation = myCSV.neerslag

listofWP = [myCSV.windkracht6,myCSV.windkracht7,myCSV.windkracht8,myCSV.windkracht9,myCSV.windkracht10,myCSV.windkracht11,myCSV.windkracht12]

def plotWindPower(duneheight, listofWP):
    currentWP = 6
    fig, ax = plt.subplots(2)
    ax[0].plot(myCSV.year, duneheight, label='duneheight')
    for wp in listofWP:
        ax[1].plot(myCSV.year, wp, label='windpower ' + str(currentWP))
        currentWP += 1
    plt.xlabel('year')
    plt.legend()
    plt.show()

plotWindPower(duneheight1, listofWP)

# Wind direction plots
fig, ax = plt.subplots(3)
ax[0].plot(myCSV.year, duneheight1, label='duneheight')
ax[1].plot(myCSV.year, north, label='north')
ax[1].plot(myCSV.year, east, label='east')
ax[1].plot(myCSV.year, south, label='south')
ax[1].plot(myCSV.year, west, label='west')
ax[2].plot(myCSV.year, northeast, label='northeast')
ax[2].plot(myCSV.year, southeast, label='southeast')
ax[2].plot(myCSV.year, southwest, label='southwest')
ax[2].plot(myCSV.year, northwest, label='northwest')
plt.xlabel('year')
plt.legend()
plt.show()

# Humidity
fig, ax = plt.subplots(2)
ax[0].scatter(myCSV.year, duneheight1, label='duneheight')
ax[1].scatter(myCSV.year, highhumidity, label='high humidity')
ax[1].scatter(myCSV.year, lowhumidity, label='low humidity')
ax[1].scatter(myCSV.year, avghumidity, label='avg humidity')
plt.show()

# Precipitation
plt.scatter(myCSV.year, duneheight1, label='duneheight')
plt.scatter(myCSV.year, precipitation, label='precipitation')
plt.show()





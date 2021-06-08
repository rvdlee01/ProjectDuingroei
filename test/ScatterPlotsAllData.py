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
'''
ax1.set_title('damped')
ax1.set_xlabel('time (s)')
ax1.set_ylabel('amplitude')
'''
currentWP = 6
fig, ax = plt.subplots(7)
count = 0
for wp in listofWP:
    ax[count].scatter(wp, duneheight1, label='duneheight')
    ax[count].set_title('Duneheight vs Windpower' + str(currentWP))
    currentWP += 1
    count += 1
    plt.xlabel('Wind power')
plt.legend()

# Wind direction plots
fig, ax = plt.subplots(8)
#ax[0].plot(myCSV.year, duneheight1, label='duneheight')
ax[0].scatter(north, duneheight1, label='north')
ax[1].scatter(east, duneheight1, label='east')
ax[2].scatter(south, duneheight1, label='south')
ax[3].scatter(west, duneheight1, label='west')
ax[4].scatter(northeast, duneheight1, label='northeast')
ax[5].scatter(southeast, duneheight1, label='southeast')
ax[6].scatter(southwest, duneheight1, label='southwest')
ax[7].scatter(northwest, duneheight1, label='northwest')
plt.xlabel('Wind direction')
plt.legend()

# Humidity
fig, ax = plt.subplots(3)
#ax[0].scatter(myCSV.year, duneheight1, label='duneheight')
ax[0].scatter(highhumidity, duneheight1, label='high humidity')
ax[1].scatter(lowhumidity, duneheight1, label='low humidity')
ax[2].scatter(avghumidity, duneheight1, label='avg humidity')

# Precipitation
plt.scatter(precipitation, duneheight1, label='duneheight')
plt.legend()
#ax[0].scatter(myCSV.year, precipitation, label='precipitation')
plt.show()





from tkinter import *
from tkinter import filedialog
import csv
import pandas as pd

# root = Tk()

# uitkiezen van een file


def SelectFile():
    file_path = filedialog.askopenfilename(
        title="select a file", filetype=[("CSV files", '.csv')])
    return file_path
    # print(file_path)
    # file_path = 0

    # uitlezen van een file
    # f = open(file_path)
    # f = open("D:/Informatica/Informatica jaar 2/Periode 4/Project/test.csv", "r")
    # data = f.read()
    # print(f)
    # print(data)
selected_file = SelectFile()
f = open(selected_file)

df = pd.read_csv(selected_file)
# print(df)
Jaar = input("Kolom naam jaar: ")
Windkracht7 = input("Windkracht7: ")
Windkracht8 = input("Windkracht8: ")
Windkracht9 = input("Windkracht9: ")
Windkracht10 = input("Windkracht10: ")
Windkracht11 = input("Windkracht11: ")
Neerslag = input("Neerslag: ")

X = df[[Jaar, Windkracht7, Windkracht8, Windkracht9 , Windkracht10, Windkracht11, Neerslag ]]
print(X)

# data = f.read()
# print(data)
# outfile = open("Weerdata.csv", 'w')
# with open(selected_file, 'r') as csv_file:
#     csv_reader = csv.reader(csv_file)
#     header = next(csv_reader)
#     for row in csv_reader:
#         jaar = row[0]
#         hoogte = row[1]
#         stormdagen = row[2]
#         line = "{},{},{}\n".format(jaar, hoogte, stormdagen)
#         outfile.write(line)
# outfile.close()


# def createCSV(filename):
#     myCSV = pd.read_csv(filename)

# row_year = input("enter which row is Year: ")
# with open('WeatherData.csv', 'w') as new_file:

# csv_writer = csv.writer(new_file)

# row_duinhoogte = input("enter which row is Year: ")
# row_stromdagen = input("enter which row is Year: ")
# row_windkracht7 = input("enter which row is Year: ")
# row_windkracht8 = input("enter which row is Year: ")
# row_windkracht9 = input("enter which row is Year: ")
# row_windkracht10 = input("enter which row is Year: ")
# row_windkracht11 = input("enter which row is Year: ")
# row_neerslag = input("enter which row is neerslag: ")
# for row in csv_reader:
#     year = row[row_year]
#     duinhoogte = row[duinhoogte]

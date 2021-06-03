from tkinter import *
from tkinter import filedialog
import csv
import pandas as pd
from io import StringIO
import numpy as np

# root = Tk()

# uitkiezen van een file


def SelectFile():
    file_path = filedialog.askopenfilename(
        title="select a file", filetype=[("CSV files", '.csv')])
    return file_path

def CheckColumnNames(checklist, csv_columns):
    for element in checklist:
        if element not in csv_columns:
            return False
    return True

def CheckRows(selected_file):
    num_rows = -1
    for row in open(selected_file):
        num_rows += 1
    if(num_rows > 10):
        print("Correct aantal rows")
    else:
        print("Te weinig rows")
#file selecteren van windows explorer
selected_file = SelectFile()
# uitlezen van csv file
df = pd.read_csv(selected_file)
# missende data aanvullen met nan
df = df.fillna(np.nan)
# df = df.astype(int)
# lijst maken van kolom namen door .columns 
list_of_column_names = list(df.columns)
check_list = ['jaar','duinhoogte','stormdagen','windkracht7','windkracht8','windkracht9','windkracht10','windkracht11','neerslag']
# elk woord in de lijst zetten naar HOOFDLETTERS
check_list = [each_string.lower() for each_string in check_list]
list_of_column_names = [each_string.lower() for each_string in list_of_column_names]
#functie aanroepen
booleanCheck = CheckColumnNames(check_list, list_of_column_names)
if(booleanCheck):
    print("Kolommen kloppen")
else:
    print("Kolommen kloppen niet")
CheckRows(selected_file) 


print(df.head())



# if(check_list == list_of_column_names):
#     print("correct buiten functie")
# print(df)
# f = open(selected_file)


# df_cols = pd.read_csv(selected_file, nrows=0)
# print("df")
# print(df_cols)
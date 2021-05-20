from tkinter import *
from tkinter import filedialog

# root = Tk()

# uitkiezen van een file
# file_path = filedialog.askopenfilename()
# print(file_path)
file_path = 0
# uitlezen van een file
f = open(file_path)
# f = open("D:/Informatica/Informatica jaar 2/Periode 4/Project/test.csv", "r")
data = f.read()
# print(f)
print(data)
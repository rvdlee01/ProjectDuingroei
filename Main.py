from tkinter.constants import DISABLED
import matplotlib
matplotlib.use("TkAgg")
import pandas as pd

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from tkinter import *
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

from tkinter import font

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

LARGE_FONT = ("Verdana", 12)
filename =''


def load_Data(self,csv_data):
    try:
        df = pd.read_csv(csv_data)
        self.tv1["column"] = list(df.columns)
        self.tv1["show"] = "headings"
        for column in self.tv1["columns"]:
            self.tv1.heading(column, text=column)
        
        df_rows = df.to_numpy().tolist()
        for row in df_rows:
            self.tv1.insert("", "end", values=row)
        self.button1["state"] = tk.NORMAL
        self.button2.configure(bg="green")
    except:
        tk.messagebox.showerror("Error", "Wrong file or file format!")
    return None



def getCsvFile(self):
    global filename
    filename = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = [("CSV files", '.csv')])
    if filename != '':
        self.tv1.delete(*self.tv1.get_children()) #Empties rows and columns in csvTable
        load_Data(self,filename)

#JASPER NIEUWE FUNCTIE VOOR NAMEN CHECK HIER :D
        

def model_NN(filename):
    df = pd.read_csv(filename)
    df=df[df['year'] > 2002]

    df['YYYYMMDD'] = df['year']
    x = df[['windkracht6','windkracht7','windkracht8','windkracht9','windkracht10','windkracht11','windkracht12','north','east','south','west','northeast','southeast','southwest','northwest','highhumidity','lowhumidity','aveghumidity','neerslag']]
    y = df['punt1'].values
    #90% training data & 10% testing data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

    #Standardize data
    xnorm = StandardScaler()
    ynorm = StandardScaler()

    x_train = xnorm.fit_transform(x_train)
    x_test = xnorm.transform(x_test)

    y_train = ynorm.fit_transform(np.array(y_train).reshape(-1,1))
    y_test = ynorm.transform(np.array(y_test).reshape(-1,1))

    #Adding dense layers
    model = Sequential()
    model.add(Dense(128, input_dim=19, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    #print(model.summary())
    model.fit(x_train,y_train, epochs=10, batch_size=50, verbose=0)

    #predictions 
    trainPredict = model.predict(x_train)
    testPredict = model.predict(x_test)

    def dataframeToCSV(y_train, y_test, filename):
        #convert np arrays to list and add the training and testing results together in a list
        listOfTraining = trainPredict.tolist()
        listOfTesting = testPredict.tolist()
        y_train, y_test = convertToList(y_train, y_test)
        joinedList = listOfTraining + listOfTesting
        joinedList2 = y_train + y_test

        #convert array(list[]) to integer
        predictedList = []
        actualList = []
        differences = []
        errorrates = []
        count = 0
        for date in df['YYYYMMDD']:
            predictedList.append(int(float(ynorm.inverse_transform(joinedList[count]))))
            actualList.append(int(float(ynorm.inverse_transform(joinedList2[count]))))
            differences.append(int(float(ynorm.inverse_transform(joinedList[count]))) - int(float(ynorm.inverse_transform(joinedList2[count]))))
            errorrates.append((int(float(ynorm.inverse_transform(joinedList[count]))) - int(float(ynorm.inverse_transform(joinedList2[count]))))
                              / int(float(ynorm.inverse_transform(joinedList2[count]))) * 100)
            count = count + 1

        #assign list to column
        newDataframe = {'Date': df['YYYYMMDD'].tolist(), 'Predicted': predictedList, 'Actual': actualList, 'Difference': differences, 'Error rate': errorrates}

        #create dataframe
        OutputDataframe = pd.DataFrame(newDataframe)

        #convert dataframe to csv
        OutputDataframe.to_csv(filename, index=False)

    dataframeToCSV(y_train, y_test, 'PredictedOutputs.csv')

    return df, y_train,y_test, ynorm.inverse_transform(y_train),ynorm.inverse_transform(trainPredict), ynorm.inverse_transform(y_test), ynorm.inverse_transform(testPredict)


class DuinGroeiApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.iconbitmap(self, default="dune_icon.ico")
        tk.Tk.wm_title(self, "DuinHoogte")
        tk.Tk.wm_geometry(self, "1280x720")
        

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        
        for F in (StartPage, PageOne):
            frame = F(container, self)
            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

    def get_page(self, page_class): # returns reference to a page
        return self.frames[page_class]

class StartPage(tk.Frame):
    global button1
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        myFont = font.Font(family = 'Helvetica', size = 30)
        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        self.button1 = tk.Button(self,state = DISABLED, text="Go to graph page",
                            command=lambda: [PageOne.plotGraph(controller.get_page(PageOne)),controller.show_frame(PageOne),self.csvTable.pack_forget()], font = myFont)
        self.button2 = tk.Button(self, text="Select csv file",
                            command=lambda: getCsvFile(self), font = myFont)
        self.button1.pack(padx=50,pady=50)
        self.button2.pack()
        
        # Frame for Treeview
        self.csvTable = tk.LabelFrame(text ="CSV data")
        self.csvTable.pack(anchor='center',ipadx=400,ipady=250, pady=25)


        ## Treeview Widget
        self.tv1 = ttk.Treeview(self.csvTable)
        self.tv1.place(relheight=1, relwidth=1)

        treescrolly = tk.Scrollbar(self.csvTable, orient= "vertical", command = self.tv1.yview)
        treescrollx = tk.Scrollbar(self.csvTable, orient= "horizontal", command = self.tv1.xview)
        self.tv1.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set)
        treescrollx.pack(side="bottom", fill="x")
        treescrolly.pack(side="right", fill="y")

        # self.csvTable.destroy()

def convertToList(arrayX, arrayY):
    y, x = [],[]
    for height in arrayY:
        y.append(height)
    for xas in arrayX:
        x.append(xas)
    return x, y

class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        start_page = self.controller.get_page(StartPage) # Access variables from another class
        label = tk.Label(self, text="Graph page", font=LARGE_FONT)
        label.pack(pady=10,padx=10)
        button1 = ttk.Button(self, text="Home",
                            command=lambda: [controller.show_frame(StartPage),self.csvTable2.pack_forget(),self.clearGraphpage(),self.canvas.get_tk_widget().pack_forget(),start_page.csvTable.pack(anchor='center',ipadx=400,ipady=250, pady=25)])
        button1.pack()

        # Frame for Treeview
        self.csvTable2 = tk.LabelFrame(text ="CSV data")


        ## Treeview Widget
        self.tv2 = ttk.Treeview(self.csvTable2)
        self.tv2.place(relheight=1, relwidth=1)

        treescrolly = tk.Scrollbar(self.csvTable2, orient= "vertical", command = self.tv2.yview)
        treescrollx = tk.Scrollbar(self.csvTable2, orient= "horizontal", command = self.tv2.xview)
        self.tv2.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set)
        treescrollx.pack(side="bottom", fill="x")
        treescrolly.pack(side="right", fill="y")

        self.f = Figure(figsize=(5,5), dpi=100)
        self.a = self.f.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.f, self)
        
    def clearGraphpage(self):
        for item in self.canvas.get_tk_widget().find_all():
            self.canvas.get_tk_widget().delete(item)


    def plotGraph(self):
        if(filename != ''):
            df,y_train,y_test,a,b,c,d = model_NN(filename)

            ax, ay = convertToList(df['YYYYMMDD'][:y_train.shape[0]], a)
            bx, by = convertToList(df['YYYYMMDD'][:y_train.shape[0]], b)
            cx, cy = convertToList(df['YYYYMMDD'][y_train.shape[0]:y_train.shape[0]+y_test.shape[0]], c)
            dx, dy = convertToList(df['YYYYMMDD'][y_train.shape[0]:y_train.shape[0]+y_test.shape[0]], d)

            self.a.clear()
            self.a.plot(ax,ay)
            self.a.plot(bx,by)
            self.a.plot(cx,cy)
            self.a.plot(dx,dy)
            
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            for i in self.tv2.get_children():
                self.tv2.delete(i)

            df = pd.read_csv('PredictedOutputs.csv')
            self.tv2["column"] = list(df.columns)
            self.tv2["show"] = "headings"
            for column in self.tv2["columns"]:
                self.tv2.heading(column, text=column)
            
            df_rows = df.to_numpy().tolist()
            for row in df_rows:
                self.tv2.insert("", "end", values=row)

            self.csvTable2.pack(anchor='center',ipadx=400,ipady=250, pady=25)



app = DuinGroeiApp()
app.minsize(800,600)
app.mainloop()

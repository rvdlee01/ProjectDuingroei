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
        

def model_NN(filename,inputX):
    df = pd.read_csv(filename)
    df=df[df['year'] > 2002]

    df['YYYYMMDD'] = df['year']
    x = df[['windkracht6','windkracht7','windkracht8','windkracht9','windkracht10','windkracht11','windkracht12','north','east','south','west','northeast','southeast','southwest','northwest','highhumidity','lowhumidity','aveghumidity','neerslag']]
    y = df['punt1'].values
    #90% training data & 10% testing data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

    x_userinput = inputX[['windkracht6','windkracht7','windkracht8','windkracht9','windkracht10','windkracht11','windkracht12','north','east','south','west','northeast','southeast','southwest','northwest','highhumidity','lowhumidity','aveghumidity','neerslag']]
    
    #Standardize data
    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_userinput = scaler.transform(x_userinput)

    y_train = scaler.fit_transform(np.array(y_train).reshape(-1,1))
    y_test = scaler.transform(np.array(y_test).reshape(-1,1))

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
    inputPredict = model.predict(x_userinput) 
    years = df['YYYYMMDD'].tolist()
    
    #convert np arrays to list and add the training and testing results together in a list
    listOfTraining = trainPredict.tolist()
    listOfTesting = testPredict.tolist()
    listOfInput = inputPredict.tolist()
    joinedList = listOfTraining + listOfTesting
    joinedList2 = []
    def dataframeToCSV(y_train, y_test, filename):
        y_train, y_test = convertToList(y_train, y_test)
        for i in y_train + y_test:
            joinedList2.append(i)
        #convert array(list[]) to integer
        predictedList = []
        actualList = []
        differences = []
        errorrates = []
        count = 0
        for date in df['YYYYMMDD']:
            predictedList.append(int(float(scaler.inverse_transform(joinedList[count]))))
            actualList.append(int(float(scaler.inverse_transform(joinedList2[count]))))
            differences.append(int(float(scaler.inverse_transform(joinedList[count]))) - int(float(scaler.inverse_transform(joinedList2[count]))))
            errorrates.append((int(float(scaler.inverse_transform(joinedList[count]))) - int(float(scaler.inverse_transform(joinedList2[count]))))
                              / int(float(scaler.inverse_transform(joinedList2[count]))) * 100)
            count = count + 1
        #add prediction of userinput to the lists
        predictedList.append(int(float(scaler.inverse_transform(listOfInput[0]))))
        actualList.append(None)
        differences.append(None)
        errorrates.append(None)
        years.append(int(inputX['year'].values))
        #assign list to column
        newDataframe = {'Date': years, 'Predicted': predictedList, 'Actual': actualList, 'Difference': differences, 'Error rate': errorrates}
        #create dataframe
        OutputDataframe = pd.DataFrame(newDataframe)
        #convert dataframe to csv
        OutputDataframe.to_csv(filename, index=False)
        print(OutputDataframe)


    dataframeToCSV(y_train, y_test, 'PredictedOutputs.csv')
    return df, y_train,y_test, scaler.inverse_transform(y_train),scaler.inverse_transform(trainPredict), scaler.inverse_transform(y_test), scaler.inverse_transform(testPredict), scaler.inverse_transform(inputPredict), years,scaler.inverse_transform(joinedList), scaler.inverse_transform(joinedList2)


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
        myFont = font.Font(family = 'Helvetica', size = 20)
        label = tk.Label(self, text="Startpagina", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        # Buttons
        self.button1 = tk.Button(self,state = DISABLED, text="Voorspellen", width=18,
                            command=lambda: [PageOne.plotGraph(controller.get_page(PageOne)),controller.show_frame(PageOne),self.csvTable.pack_forget()], font = myFont)
        self.button2 = tk.Button(self, text="CSV bestand selecteren", width=18,
                            command=lambda: getCsvFile(self), font = myFont)
        self.button2.pack(ipadx=5,ipady=3,padx=30,pady=25, side=LEFT)#anchor="ne"
        self.button1.pack(ipadx=5,ipady=3,padx=30,pady=15, side=LEFT)#anchor="ne"

        listOfInputVariables = ['year','wp6','wp7','wp8','wp9','wp10','wp11','wp12','north','east','south','west','northeast','southeast','southwest','northwest','highhumidity','lowhumidity','avghumidity','precipitation']
        dictOfDirections = {'north':'noorden','east':'oosten','south':'zuiden','west':'westen','northeast':'noord-oosten','southeast':'zuid-oosten','southwest':'zuid-westen','northwest':'noord-westen'}
        dictOfHumidity = {'highhumidity': 'hoge luchtvochtigheid','lowhumidity':'lage luchtvochtigheid','avghumidity': 'gemiddelde luchtvochtigheid'}
        count = 6
        for value in listOfInputVariables:
            # Variables for storing data to predict dune height
            setattr(self,value,tk.StringVar())
            if value == 'year':
                # Year entry box and label
                setattr(self,value+'_label',tk.Label(self, text = 'Jaar', font=('calibre',10, 'bold')).pack(padx=30,anchor="ne"))
                setattr(self,'entry'+value,tk.Entry(self, width=25, textvariable = self.year).pack(padx=30,pady=5,anchor="ne"))
            elif 'wp' in value:
                # Wind power entry boxes and labels
                setattr(self,value+'_label',tk.Label(self, text = 'Aantal dagen met windkracht ' + str(count), font=('calibre',10, 'bold')).pack(padx=30,anchor="ne"))
                setattr(self,'entry'+value,tk.Entry(self, width=25, textvariable = getattr(self,value)).pack(padx=30,anchor="ne")) #pady=5
                count += 1
            elif value in dictOfDirections.keys():
                # Wind direction entry boxes and labels
                for k, v in dictOfDirections.items():
                    if k == value:
                        textInputWD = 'Aantal dagen met wind vanuit het ' + v
                setattr(self,value+'_label',tk.Label(self, text = textInputWD, font=('calibre',10, 'bold')).pack(padx=30,anchor="ne"))
                setattr(self,'entry'+value,tk.Entry(self, width=25, textvariable = getattr(self,value)).pack(padx=30,anchor="ne")) #pady=5
            elif value in dictOfHumidity.keys():
                # Humidity entry boxes and labels
                for k, v in dictOfHumidity.items():
                    if k == value:
                        textInputH = 'Aantal dagen met een ' + v
                setattr(self,value+'_label',tk.Label(self, text = textInputH, font=('calibre',10, 'bold')).pack(padx=30,anchor="ne"))
                setattr(self,'entry'+value,tk.Entry(self, width=25, textvariable = getattr(self,value)).pack(padx=30,anchor="ne")) #pady=5
            elif value == 'precipitation':
                # Precipitation entry box and label
                setattr(self,value+'_label',tk.Label(self, text = 'Neerslag in een jaar', font=('calibre',10, 'bold')).pack(padx=30,anchor="ne"))
                setattr(self,'entry'+value,tk.Entry(self, width=25, textvariable = getattr(self,value)).pack(padx=30,anchor="ne")) #pady=5
        
        # Frame for Treeview
        self.csvTable = tk.LabelFrame(text ="CSV data")
        self.csvTable.pack(padx=30,pady=25,ipadx=400,ipady=250,anchor='nw')

        # Treeview Widget
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
        self.start_page = self.controller.get_page(StartPage) # Access variables from another class
        label = tk.Label(self, text="Grafiekpagina", font=LARGE_FONT)
        label.pack(pady=10,padx=10)
        button1 = ttk.Button(self, text="Terug naar start",
                            command=lambda: [controller.show_frame(StartPage),self.csvTable2.pack_forget(),self.clearGraphpage(),self.canvas.get_tk_widget().pack_forget(),self.start_page.csvTable.pack(anchor='center',ipadx=400,ipady=250, pady=25)])
        button1.pack()
        
        # Frame for Treeview
        self.csvTable2 = tk.LabelFrame(text ="CSV data")


        # Treeview Widget
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
            start_page = self.start_page
            inputX = pd.DataFrame(columns=['year','windkracht6','windkracht7','windkracht8','windkracht9','windkracht10','windkracht11','windkracht12','north','east','south','west','northeast','southeast','southwest','northwest','highhumidity','lowhumidity','aveghumidity','neerslag'])
            inputX.loc[0] = [start_page.year.get(),start_page.wp6.get(),start_page.wp7.get(),start_page.wp8.get(),start_page.wp9.get(),start_page.wp10.get(),start_page.wp11.get(),start_page.wp12.get(),start_page.north.get(),start_page.east.get(),
                      start_page.south.get(),start_page.west.get(),start_page.northeast.get(),start_page.southeast.get(),start_page.southwest.get(),start_page.northwest.get(),start_page.highhumidity.get(),start_page.lowhumidity.get()
                      ,start_page.avghumidity.get(),start_page.precipitation.get()]
            
            df,y_train,y_test,a,b,c,d,userOutput,years,listoftraining,listoftesting = model_NN(filename,inputX)
            print('prediction of userinput: ', userOutput)
            ax, ay = convertToList(years[:y_train.shape[0]], a)
            bx, by = convertToList(years[:y_train.shape[0]], b)
            cx, cy = convertToList(years[y_train.shape[0]:y_train.shape[0]+y_test.shape[0]], c)
            dx, dy = convertToList(years[y_train.shape[0]:y_train.shape[0]+y_test.shape[0]], d)
            userOutputx, userOutputy = convertToList(years[y_train.shape[0]+y_test.shape[0]:y_train.shape[0]+y_test.shape[0]+1], userOutput)
            trainingx, trainingy = convertToList(years[:y_train.shape[0]+y_test.shape[0]], listoftraining)
            testingx, testingy = convertToList(years[:y_train.shape[0]+y_test.shape[0]], listoftesting)

            self.a.clear()
            self.a.plot(trainingx,trainingy,label="predicted values")
            self.a.plot(testingx,testingy,label="actual values")
            self.a.plot(ax,ay,label="actual training values")
            self.a.plot(bx,by,label="predicted training values")
            self.a.plot(cx,cy,label="actual testing values")
            self.a.plot(dx,dy,label="predicted testing values")
            self.a.scatter(userOutputx,userOutputy,label="predicted user input")

            self.a.legend()
            
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

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
    model.add(Dense(1, activation='relu')) #maybe sigmoid? max 3 layers!
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

    OutputDataframe = dataframeToCSV(y_train, y_test,joinedList2,joinedList,df,scaler,years,listOfInput,inputX)
    return df, y_train,y_test, scaler.inverse_transform(y_train),scaler.inverse_transform(trainPredict), scaler.inverse_transform(y_test), scaler.inverse_transform(testPredict), scaler.inverse_transform(inputPredict), years,scaler.inverse_transform(joinedList), scaler.inverse_transform(joinedList2),OutputDataframe

def dataframeToCSV(y_train, y_test,joinedList2,joinedList,df,scaler,years,listOfInput,inputX):
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
    print(OutputDataframe)
    return OutputDataframe

def convertToList(arrayX, arrayY):
    y, x = [],[]
    for height in arrayY:
        y.append(height)
    for xas in arrayX:
        x.append(xas)
    return x, y


class Mainscreen(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        main_frame = Frame(container)
        main_frame.pack(fill=BOTH, expand=1)

        my_canvas = Canvas(main_frame)
        my_canvas.pack(side=LEFT, fill=BOTH, expand=1)

        my_scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=my_canvas.yview)
        my_scrollbar.pack(side=RIGHT, fill=Y)

        my_canvas.configure(yscrollcommand=my_scrollbar.set)
    
        my_canvas.bind(
                "<Configure>",
                lambda e: my_canvas.configure(
                    scrollregion=my_canvas.bbox("all")
                )
            )

        second_frame = Frame(my_canvas)

        my_canvas.create_window((0,0), window=second_frame, anchor="nw")

        predictbutton = Button(second_frame,state = DISABLED, text="Voorspellen", width=18,
                            command=lambda: [my_canvas.pack_forget(), GraphPage(container, self),my_scrollbar.pack_forget(),main_frame.pack_forget()])
        predictbutton.grid(row=0,column=1,padx=10,pady=10)
        uploadbutton = Button(second_frame, text="CSV bestand selecteren", width=18,
                            command=lambda: getCsvFile(uploadbutton, predictbutton, tv1))
        uploadbutton.grid(row=1,column=1,padx=10,pady=10)
        
        listOfInputVariables = ['year','wp6','wp7','wp8','wp9','wp10','wp11','wp12','north','east','south','west','northeast','southeast','southwest','northwest','highhumidity','lowhumidity','avghumidity','precipitation']
        dictOfDirections = {'north':'noorden','east':'oosten','south':'zuiden','west':'westen','northeast':'noord-oosten','southeast':'zuid-oosten','southwest':'zuid-westen','northwest':'noord-westen'}
        dictOfHumidity = {'highhumidity': 'hoge luchtvochtigheid','lowhumidity':'lage luchtvochtigheid','avghumidity': 'gemiddelde luchtvochtigheid'}
        count = 6
        for value in listOfInputVariables:
            # Variables for storing data to predict dune height
            setattr(self,value,StringVar())
            if value == 'year':
                # Year entry box and label
                setattr(self,value+'_label',Label(second_frame, text = 'Jaar', font=('calibre',10, 'bold')).grid(padx=30,column=1,sticky="ne"))
                setattr(self,'entry'+value,Entry(second_frame, width=25, textvariable = self.year).grid(padx=30,pady=5,column=1,sticky="ne"))
            elif 'wp' in value:
                # Wind power entry boxes and labels
                setattr(self,value+'_label',Label(second_frame, text = 'Aantal dagen met windkracht ' + str(count), font=('calibre',10, 'bold')).grid(padx=30,column=1,sticky="ne"))
                setattr(self,'entry'+value,Entry(second_frame, width=25, textvariable = getattr(self,value)).grid(padx=30,column=1,sticky="ne")) #pady=5
                count += 1
            elif value in dictOfDirections.keys():
                # Wind direction entry boxes and labels
                for k, v in dictOfDirections.items():
                    if k == value:
                        textInputWD = 'Aantal dagen met wind vanuit het ' + v
                setattr(self,value+'_label',Label(second_frame, text = textInputWD, font=('calibre',10, 'bold')).grid(padx=30,column=1,sticky="ne"))
                setattr(self,'entry'+value,Entry(second_frame, width=25, textvariable = getattr(self,value)).grid(padx=30,column=1,sticky="ne")) #pady=5
            elif value in dictOfHumidity.keys():
                # Humidity entry boxes and labels
                for k, v in dictOfHumidity.items():
                    if k == value:
                        textInputH = 'Aantal dagen met een ' + v
                setattr(self,value+'_label',Label(second_frame, text = textInputH, font=('calibre',10, 'bold')).grid(padx=30,column=1,sticky="ne"))
                setattr(self,'entry'+value,Entry(second_frame, width=25, textvariable = getattr(self,value)).grid(padx=30,column=1,sticky="ne")) #pady=5
            elif value == 'precipitation':
                # Precipitation entry box and label
                setattr(self,value+'_label',Label(second_frame, text = 'Neerslag in een jaar', font=('calibre',10, 'bold')).grid(padx=30,column=1,sticky="ne"))
                setattr(self,'entry'+value,Entry(second_frame, width=25, textvariable = getattr(self,value)).grid(padx=30,column=1,sticky="ne")) #pady=5
        
        # Frame for Treeview
        csvTable = LabelFrame(second_frame,text ="CSV data")
        csvTable.grid(padx=30,pady=25,ipadx=400,ipady=250,row=41,column=1)

        # Treeview Widget
        tv1 = ttk.Treeview(csvTable)
        tv1.place(relheight=1, relwidth=1)

        treescrolly = tk.Scrollbar(csvTable, orient= "vertical", command = tv1.yview)
        treescrollx = tk.Scrollbar(csvTable, orient= "horizontal", command = tv1.xview)
        tv1.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set)
        treescrollx.pack(side="bottom", fill="x")
        treescrolly.pack(side="right", fill="y")

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
                return True
            else:
                print("Te weinig rows")
                return False

        def load_Data(tv1,csv_data):
            try:
                df = pd.read_csv(csv_data)
                tv1["column"] = list(df.columns)
                tv1["show"] = "headings"
                for column in tv1["columns"]:
                    tv1.heading(column, text=column)
                
                df_rows = df.to_numpy().tolist()
                for row in df_rows:
                    tv1.insert("", "end", values=row)

                # missende data aanvullen met nan
                df = df.fillna(np.nan)
                # lijst maken van kolom namen door .columns 
                list_of_column_names = list(df.columns)
                check_list = ['year','punt1','punt2','punt3','windkracht6','windkracht7','windkracht8','windkracht9','windkracht10','windkracht11','windkracht12','north','east','south','west','northeast','southeast','southwest','northwest','highhumidity','lowhumidity','aveghumidity','neerslag']
                # elk woord in de lijst zetten naar HOOFDLETTERS
                list_of_column_names = [each_string.lower() for each_string in list_of_column_names]
                #functie aanroepen
                boolColumns = CheckColumnNames(check_list, list_of_column_names)
                if(boolColumns):
                    print("Kolommen kloppen")
                else:
                    print("Kolommen kloppen niet")
                boolRows = CheckRows(csv_data)
                return boolColumns, boolRows
            except:
                tk.messagebox.showerror("Error", "Wrong file or file format!")
            return None

        def getCsvFile(uploadbutton, predictbutton, tv1): #treeview, buttons
            global filename
            filename = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = [("CSV files", '.csv')])
            if filename != '':
                boolColumns, boolRows = load_Data(tv1,filename)
                if ((boolColumns == False) or (boolRows == False)):
                    predictbutton["state"] = DISABLED
                    uploadbutton.configure(bg="red")
                if ((boolColumns == True) and (boolRows == True)):
                    predictbutton["state"] = NORMAL
                    uploadbutton.configure(bg="green")

def clearGraphpage(canvas):
    for item in canvas.get_tk_widget().find_all():
        canvas.get_tk_widget().delete(item)


def plotGraph(a,f,canvas,startpage,tv2,csvTable2):
    start_page = startpage
    inputX = pd.DataFrame(columns=['year','windkracht6','windkracht7','windkracht8','windkracht9','windkracht10','windkracht11','windkracht12','north','east','south','west','northeast','southeast','southwest','northwest','highhumidity','lowhumidity','aveghumidity','neerslag'])
    inputX.loc[0] = [start_page.year.get(),start_page.wp6.get(),start_page.wp7.get(),start_page.wp8.get(),start_page.wp9.get(),start_page.wp10.get(),start_page.wp11.get(),start_page.wp12.get(),start_page.north.get(),start_page.east.get(),
                start_page.south.get(),start_page.west.get(),start_page.northeast.get(),start_page.southeast.get(),start_page.southwest.get(),start_page.northwest.get(),start_page.highhumidity.get(),start_page.lowhumidity.get()
                ,start_page.avghumidity.get(),start_page.precipitation.get()]
            
    df,y_train,y_test,train_actual,train_predict,test_actual,test_predict,userOutput,years,listoftraining,listoftesting,OutputDataframe = model_NN(filename,inputX)
    print('prediction of userinput: ', userOutput)
    ax, ay = convertToList(years[:y_train.shape[0]], train_actual)
    bx, by = convertToList(years[:y_train.shape[0]], train_predict)
    cx, cy = convertToList(years[y_train.shape[0]:y_train.shape[0]+y_test.shape[0]], test_actual)
    dx, dy = convertToList(years[y_train.shape[0]:y_train.shape[0]+y_test.shape[0]], test_predict)
    userOutputx, userOutputy = convertToList(years[y_train.shape[0]+y_test.shape[0]:y_train.shape[0]+y_test.shape[0]+1], userOutput)
    trainingx, trainingy = convertToList(years[:y_train.shape[0]+y_test.shape[0]], listoftraining)
    testingx, testingy = convertToList(years[:y_train.shape[0]+y_test.shape[0]], listoftesting)

    a.clear()
    a.plot(trainingx,trainingy,label="predicted values")
    a.plot(testingx,testingy,label="actual values")
    a.plot(ax,ay,label="actual training values")
    a.plot(bx,by,label="predicted training values")
    a.plot(cx,cy,label="actual testing values")
    a.plot(dx,dy,label="predicted testing values")
    a.scatter(userOutputx,userOutputy,label="predicted user input")

    a.legend()
    canvas.draw()
    canvas.get_tk_widget().grid(sticky="n", column=2,row=1)

    for i in tv2.get_children():
        tv2.delete(i)

    tv2["column"] = list(df.columns)
    tv2["show"] = "headings"
    for column in tv2["columns"]:
        tv2.heading(column, text=column)
            
    df_rows = df.to_numpy().tolist()
    for row in df_rows:
        tv2.insert("", "end", values=row)

    csvTable2.grid(sticky='n',ipadx=400,ipady=250, pady=25,column=1,row=1)

    return OutputDataframe

class GraphPage(ttk.Frame):
    def __init__(self, container, start_page, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        plot_frame = Frame(container)
        plot_frame.pack(fill=BOTH, expand=1)

        my_canvas = Canvas(plot_frame)
        my_canvas.pack(side=LEFT, fill=BOTH, expand=1)

        my_scrollbar = ttk.Scrollbar(plot_frame, orient=VERTICAL, command=my_canvas.yview)
        my_scrollbar.pack(side=RIGHT, fill=Y)

        my_canvas.configure(yscrollcommand=my_scrollbar.set)
    
        my_canvas.bind(
                "<Configure>",
                lambda e: my_canvas.configure(
                    scrollregion=my_canvas.bbox("all")
                )
            )

        second_frame = Frame(my_canvas)

        my_canvas.create_window((0,0), window=second_frame, anchor="nw")

        f = Figure(figsize=(5,5), dpi=100)
        a = f.add_subplot(111)
        canvas = FigureCanvasTkAgg(f, second_frame)

        # Frame for Treeview
        csvTable2 = LabelFrame(second_frame,text ="CSV data")
        csvTable2.grid(padx=30,pady=25,ipadx=400,ipady=250,row=2,column=1)

        # Treeview Widget
        tv2 = ttk.Treeview(csvTable2)
        tv2.place(relheight=1, relwidth=1)

        treescrolly = Scrollbar(csvTable2, orient= "vertical", command = tv2.yview)
        treescrollx = Scrollbar(csvTable2, orient= "horizontal", command = tv2.xview)
        tv2.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set)
        treescrollx.pack(side="bottom", fill="x")
        treescrolly.pack(side="right", fill="y")

        OutputDataframe = plotGraph(a,f,canvas,start_page,tv2,csvTable2)

        homebutton = Button(second_frame,state = NORMAL, text="Terug naar startpagina", width=18,
                            command=lambda: [my_canvas.pack_forget(),csvTable2.grid_forget(),canvas.get_tk_widget().pack_forget(),clearGraphpage(canvas),my_scrollbar.pack_forget(),plot_frame.pack_forget(),Mainscreen(container)])
        homebutton.grid(row=0,column=1)

        downloadgraph = Button(second_frame,state = NORMAL, text="Download grafiek", width=18,
                            command=lambda: [f.set_figheight(10),f.set_figwidth(20),a.get_figure().savefig('downloads/duingroeivoorspelling.png')])
        downloadgraph.grid(sticky="e",row=0,column=2)

        downloadcsv = Button(second_frame,state = NORMAL, text="Download csv", width=18,
                            command=lambda: [OutputDataframe.to_csv('downloads/PredictedOutputs.csv', index=False)])
        downloadcsv.grid(sticky="w",row=0,column=2)
            
def main():
    root = tk.Tk()
    root.geometry('1200x800')

    Mainscreen(root)
    
    root.minsize(900,700)
    root.mainloop()
    
main()

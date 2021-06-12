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

import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

filename =''
backgroundcolor="floral white"
labelfont = ('calibre',10, 'bold')
xcolumnnames = ['windkracht6','windkracht7','windkracht8','windkracht9','windkracht10','windkracht11','windkracht12','noord','oost','zuid','west','noord-oosten','zuid-oosten','zuid-westen','noord-westen','hogeluchtvochtigheid',
                'lageluchtvochtigheid','gemiddeldeluchtvochtigheid','neerslag']
ycolumnname = 'duinhoogte'

def detect_outlier(data):
    outliers = []
    threshold = 3
    mean = np.mean(data)
    std = np.std(data)
    
    for y in data:
        z_score = (y - mean)/std 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

def model_NN(filename,inputX):
    df = pd.read_csv(filename)
    df = df[df['jaar'] > 2002] #TEMP

    x = df[xcolumnnames]
    y = df[ycolumnname].values

    x_userinput = inputX[xcolumnnames]
    
    #Standardize data
    scaler = StandardScaler()

    x_dataset = scaler.fit_transform(x)
    x_userinput = scaler.transform(x_userinput)
    
    y_dataset = scaler.fit_transform(np.array(y).reshape(-1,1))

    model = keras.models.load_model('model')
    
    #predictions 
    datasetPredict = model.predict(x_dataset)
    inputPredict = model.predict(x_userinput) 
    years = df['jaar'].tolist()
    
    #convert np arrays to list and add the training and testing results together in a list
    listOfDataset = datasetPredict.tolist()
    listOfInput = inputPredict.tolist()

    OutputDataframe = dataframeToCSV(y_dataset, listOfDataset, listOfInput, df, scaler, years, inputX,inputPredict)
    return df, scaler.inverse_transform(y_dataset), scaler.inverse_transform(listOfDataset), scaler.inverse_transform(inputPredict), years, OutputDataframe

def dataframeToCSV(y_dataset, listOfDataset, listOfInput, df, scaler, years, inputX,inputPredict):
    predictedList = []
    actualList = []
    differences = []
    errorrates = []
    count = 0
    for date in df['jaar']:
        predictedList.append(int(float(scaler.inverse_transform(listOfDataset[count]))))
        actualList.append(int(float(scaler.inverse_transform(y_dataset[count]))))
        differences.append(int(float(scaler.inverse_transform(listOfDataset[count]))) - int(float(scaler.inverse_transform(y_dataset[count]))))
        errorrates.append((int(float(scaler.inverse_transform(listOfDataset[count]))) - int(float(scaler.inverse_transform(y_dataset[count]))))
                            / int(float(scaler.inverse_transform(y_dataset[count]))) * 100)
        count = count + 1
    #add prediction of userinput to the lists
    predictedList.append(int(float(scaler.inverse_transform(listOfInput[0]))))
    actualList.append(None)
    differences.append(None)
    errorrates.append(None)
    years.append(int(inputX['jaar'].values))
    #assign list to column
    newDataframe = {'Jaar': years,'Voorspelling': predictedList, 'Daadwerkelijk': actualList, 'Verschil': differences, 'Foutpercentage': errorrates}
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

        my_canvas.bind_all("<MouseWheel>",lambda event: my_canvas.yview_scroll(int(-1*(event.delta/120)), "units"))

        my_scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=my_canvas.yview)
        my_scrollbar.pack(side=RIGHT, fill=Y)

        my_canvas.configure(yscrollcommand=my_scrollbar.set)
    
        my_canvas.bind(
                "<Configure>",
                lambda e: my_canvas.configure(
                    scrollregion=my_canvas.bbox("all")
                )
            )

        second_frame = Frame(my_canvas, bg=backgroundcolor)

        my_canvas.create_window((0,0), window=second_frame, anchor = "nw")

        predictbutton = Button(second_frame, text="Voorspellen", width=18, bg="snow",
                            command=lambda: checkInputs())
        predictbutton.grid(row=0,column=2,padx=10,pady=15)
        uploadbutton = Button(second_frame, text="CSV bestand selecteren", width=18, bg="snow",
                            command=lambda: getCsvFile(uploadbutton, predictbutton, tv1))
        uploadbutton.grid(row=0,column=1,padx=10,pady=15)

        #Close help page window
        def on_closing():
            self.helppage.destroy()
            self.helppageactived = False

        #Create help page window
        def helppage(root):
            if self.helppageactived == False:
                helpWindow = Toplevel(root)
                helpWindow.title("Hulppagina")
                helpWindow.geometry("700x600")
                Label(helpWindow,text ="Tekst voor hulppagina komt hier.").pack()
                self.helppageactived = True
                self.helppage = helpWindow
                helpWindow.protocol("WM_DELETE_WINDOW", on_closing)
            else:
                #flash window and activate bell sound if help page is already opened
                self.helppage.focus_force()
                self.helppage.bell()

        self.helppageactived = False
        self.helppage = None
        helpbutton = Button(second_frame, text ="Hulp nodig?", bg="snow", command = lambda: [helppage(container)])
        helpbutton.grid(row=0,column=3,padx=10,pady=15)
        
        listOfInputVariables = ['year','wp6','wp7','wp8','wp9','wp10','wp11','wp12','north','east','south','west','northeast','southeast','southwest','northwest','highhumidity','lowhumidity','avghumidity','precipitation']
        listOfWP = ['wp6','wp7','wp8','wp9','wp10','wp11','wp12']
        dictOfDirections = {'north':'noorden','east':'oosten','south':'zuiden','west':'westen','northeast':'noord-oosten','southeast':'zuid-oosten','southwest':'zuid-westen','northwest':'noord-westen'}
        dictOfHumidity = {'highhumidity': 'hoge luchtvochtigheid','lowhumidity':'lage luchtvochtigheid','avghumidity': 'gemiddelde luchtvochtigheid'}
        count = 6

        def countData(listData,var):
            sumData = 0
            copyList = listData.copy()
            if type(copyList) is list:
                copyList.remove(var)
            else:
                del copyList[var]
            for value in copyList:
                if getattr(self,value).get() != '':
                    sumData += int(getattr(self,value).get())
            return sumData

        # Validates if input is an integer
        def validateInput(var,P):
            total = 0
            if 'wp' in var:
                total = countData(listOfWP,var)
            if var in dictOfDirections.keys():
                total = countData(list(dictOfDirections.keys()),var)
            if var in dictOfHumidity.keys():
                total = countData(list(dictOfHumidity.keys()),var)
            if P != '' and var != 'precipitation' and str.isdigit(P):
                sumTotal = int(P) + total
                
            if (var == 'precipitation'):
                try:
                    P == '' or float(P) >= 0
                    return True
                except:
                    messagebox.showerror("Error", "Vul een geldig numerieke waarde in")
                    return False
            if (str.isdigit(P) and (sumTotal <= 365 and sumTotal >= 0)) or P == '':
                return True
            else:
                if  str.isdigit(P) == False:         
                    message = "Vul een geldig numerieke waarde in.\nDe maximale waarde is 365.\nDe minimale waarde is 0."
                elif 'wp' in var:
                    message = "De maximale waarde van alle windkracht velden is 365."
                elif var in dictOfDirections.keys():
                    message = "De maximale waarde van alle windrichting velden is 365."
                elif var in dictOfHumidity.keys():
                    message = "De maximale waarde van alle luchtvochtigheidsvelden is 365."                
                messagebox.showerror("Error", message)

                return False
            
        vcmd = self.register(validateInput)

        rownumber = 2
        
        for value in listOfInputVariables:
            # Variables for storing data to predict dune height
            setattr(self,value,StringVar())
            if value == 'year':
                # Year entry box and label
                setattr(self,value+'_label',Label(second_frame, text = 'Jaar', font=labelfont, bg=backgroundcolor).grid(padx=30,row=rownumber,column=0,sticky="e"))
            elif 'wp' in value:
                # Wind power entry boxes and labels
                setattr(self,value+'_label',Label(second_frame, text = 'Aantal dagen met windkracht ' + str(count), font=labelfont, bg=backgroundcolor).grid(padx=30,row=rownumber,column=0,sticky="e"))
                count += 1
                columnnumber = 1
            elif value in dictOfDirections.keys():
                # Wind direction entry boxes and labels
                for k, v in dictOfDirections.items():
                    if k == value:
                        textInputWD = 'Aantal dagen met wind vanuit het ' + v
                setattr(self,value+'_label',Label(second_frame, text = textInputWD, font=labelfont, bg=backgroundcolor).grid(padx=30,row=rownumber,column=2,sticky="e"))
                columnnumber = 3
            elif value in dictOfHumidity.keys():
                # Humidity entry boxes and labels
                for k, v in dictOfHumidity.items():
                    if k == value:
                        textInputH = 'Aantal dagen met een ' + v
                setattr(self,value+'_label',Label(second_frame, text = textInputH, font=labelfont, bg=backgroundcolor).grid(padx=30,row=rownumber,column=4,sticky="e"))
                columnnumber = 5
            elif value == 'precipitation':
                # Precipitation entry box and label
                setattr(self,value+'_label',Label(second_frame, text = 'Neerslag in een jaar', font=labelfont, bg=backgroundcolor).grid(padx=30,row=rownumber,column=4,sticky="e"))
                columnnumber = 5
            if value != 'year':
                inputfield = Entry(second_frame, textvariable = getattr(self,value),validate='key', validatecommand=(vcmd,value,'%P'), state=DISABLED)
                inputfield.grid(row=rownumber,column=columnnumber,pady=5,sticky="w")
                setattr(self,'entry'+value,inputfield)
            else:
                setattr(self,value,StringVar(second_frame))
                getattr(self,value).set(2021) # default value
                selectbox = OptionMenu(second_frame, getattr(self,value), 2021)
                selectbox.grid(ipadx=30,row=rownumber, column=1,sticky="w")
                selectbox['state'] = DISABLED
                setattr(self,'entry'+value,selectbox)
            rownumber += 1
            if value == 'wp12' or value == 'northwest':
                rownumber = 2
        
        # Frame for Treeview
        csvTable = LabelFrame(second_frame,text="CSV data", bg=backgroundcolor)
        csvTable.grid(padx=30,pady=25,ipadx=400,ipady=250,row=41,columnspan=6)

        # Treeview Widget
        tv1 = ttk.Treeview(csvTable)
        tv1.place(relheight=1, relwidth=1)

        treescrolly = tk.Scrollbar(csvTable, orient="vertical", command=tv1.yview)
        treescrollx = tk.Scrollbar(csvTable, orient="horizontal", command=tv1.xview)
        tv1.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set)
        treescrollx.pack(side="bottom", fill="x")
        treescrolly.pack(side="right", fill="y")

        def checkInputs():
            boolInputs = True
            for inputfield in listOfInputVariables:
                if(inputfield != 'year'):
                    if(len(getattr(self, 'entry'+inputfield).get()) == 0):
                        boolInputs = False
                        break
            sumOfWP,sumOfDirections,sumOfHumidity = 0,0,0
            allCategories = listOfWP,list(dictOfDirections.keys()),list(dictOfHumidity.keys())
            for categorie in allCategories: #wind power / wind direction / humidity
                for label in categorie: #wind power 6 etc. / north etc. / low etc.
                    if(getattr(self,label).get() != ''):
                        if(label in listOfWP):
                            sumOfWP += int(getattr(self,label).get())
                        elif(label in list(dictOfDirections.keys())):
                            sumOfDirections += int(getattr(self,label).get())
                        elif(label in list(dictOfHumidity.keys())):
                            sumOfHumidity += int(getattr(self,label).get())
            if(boolInputs and (sumOfWP == sumOfDirections and sumOfWP == sumOfHumidity)):
                my_canvas.pack_forget(), GraphPage(container, self),my_scrollbar.pack_forget(),main_frame.pack_forget()
            else:
                if(filename == ''):
                    messagebox.showerror("Error", "Upload eerst een CSV bestand en vul vervolgens alle velden in!")
                elif(boolInputs == False):
                    messagebox.showerror("Error", "Vul eerst alle velden in!")
                elif(sumOfWP != sumOfDirections or sumOfWP != sumOfHumidity): #moet naar elif veranderd worden voor total days check
                    messagebox.showerror("Error", "De totaal aantal dagen komen niet met elkaar overeen!\nTotaal aantal dagen bij windkrachten: " + str(sumOfWP)
                                         + "\nTotaal aantal dagen bij windrichtingen: " + str(sumOfDirections) + "\nTotaal aantal dagen bij luchtvochtigheid: " + str(sumOfHumidity))

        def checkColumnNames(checklist, csv_columns):
            for element in checklist:
                if element not in csv_columns:
                    print("Kolommen kloppen niet")
                    return False
            #print("Kolommen kloppen")
            return True

        def checkRows(selected_file):
            num_rows = -1
            for row in open(selected_file):
                num_rows += 1
            if(num_rows > 10):
                #print("Correct aantal rows")
                return True
            else:
                print("Te weinig rows")
                return False

        def checkValues(df):
            hasEmptyValues = df.isnull().values.any()
            if(hasEmptyValues):
                print("De geüploadde CSV bestand ontbreekt gegevens")
                return False
            else:
                #print("De geüploadde CSV bestand ontbreekt geen gegevens")
                return True

        def load_Data(tv1,csv_data):
            try:
                df = pd.read_csv(csv_data)
                #clear treeview
                for row in tv1.get_children():
                    tv1.delete(row)
                tv1["column"] = list(df.columns)
                tv1["show"] = "headings"
                for column in tv1["columns"]:
                    tv1.heading(column, text=column)
                
                df_rows = df.to_numpy().tolist()
                for row in df_rows:
                    tv1.insert("", "end", values=row)

                # list with column names of user
                list_of_column_names = list(df.columns)
                check_list = ['jaar', ycolumnname] + xcolumnnames
                # make list of column names case insensitive
                list_of_column_names = [each_string.lower() for each_string in list_of_column_names]
                boolColumns = checkColumnNames(check_list, list_of_column_names)
                boolRows = checkRows(csv_data)
                boolValues = checkValues(df)
                #detect outliers
                #outliers = detect_outlier(df['duinhoogte'])
                #print('outliers: ', outliers)
                return boolColumns, boolRows, boolValues
            except:
                tk.messagebox.showerror("Error", "Wrong file or file format!")
            return None

        def getCsvFile(uploadbutton, predictbutton, tv1): #treeview, buttons
            global filename
            filename = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = [("CSV files", '.csv')])
            if filename != '':
                boolColumns, boolRows, boolValues = load_Data(tv1,filename)
                if ((boolColumns == False) or (boolRows == False) or (boolValues == False)):
                    for inputfield in listOfInputVariables:
                        getattr(self, 'entry'+inputfield)["state"] = DISABLED
                    uploadbutton.configure(bg="red")
                    csvErrorMessage = ""
                    if(boolColumns == False):
                        csvErrorMessage += "Kolommen kloppen niet!\n"
                    if(boolRows == False):
                        csvErrorMessage += "Te weinig rows!\n"
                    if(boolValues == False):
                        csvErrorMessage += "Er ontbreken gegevens!"
                    messagebox.showerror("Error", csvErrorMessage)
                if ((boolColumns == True) and (boolRows == True) and (boolValues == True)):
                    for inputfield in listOfInputVariables:
                        getattr(self, 'entry'+inputfield)["state"] = NORMAL
                    uploadbutton.configure(bg="green")

def clearGraphpage(canvas):
    for item in canvas.get_tk_widget().find_all():
        canvas.get_tk_widget().delete(item)


def plotGraph(a,f,canvas,startpage,tv2,csvTable2):
    start_page = startpage
    inputX = pd.DataFrame(columns=['jaar']+xcolumnnames)
    inputX.loc[0] = [start_page.year.get(),start_page.wp6.get(),start_page.wp7.get(),start_page.wp8.get(),start_page.wp9.get(),start_page.wp10.get(),start_page.wp11.get(),start_page.wp12.get(),start_page.north.get(),start_page.east.get(),
                start_page.south.get(),start_page.west.get(),start_page.northeast.get(),start_page.southeast.get(),start_page.southwest.get(),start_page.northwest.get(),start_page.highhumidity.get(),start_page.lowhumidity.get()
                ,start_page.avghumidity.get(),start_page.precipitation.get()]
            
    df, y_dataset, listOfDataset, inputPredict, years, OutputDataframe = model_NN(filename,inputX)
    print('prediction of userinput: ', inputPredict)
    datasetActualx, datasetActualy = convertToList(years[:listOfDataset.shape[0]], y_dataset)
    datasetPredictedx, datasetPredictedy = convertToList(years[:listOfDataset.shape[0]], listOfDataset)
    userOutputx, userOutputy = convertToList(years[listOfDataset.shape[0]:listOfDataset.shape[0]+1], inputPredict)

    a.clear()
    a.plot(datasetPredictedx, datasetPredictedy,label="dataset predicted values",color='green')
    a.plot(datasetActualx, datasetActualy,label="dataset actual values",color='purple')
    a.scatter(userOutputx,userOutputy,label="predicted user input",color='blue')

    a.legend()
    
    canvas.draw()
    canvas.get_tk_widget().grid(sticky="n", column=1,row=2)

    for i in tv2.get_children():
        tv2.delete(i)

    tv2["column"] = list(OutputDataframe.columns)
    tv2["show"] = "headings"
    for column in tv2["columns"]:
        tv2.heading(column, text=column)
            
    df_rows = OutputDataframe.to_numpy().tolist()
    for row in df_rows:
        tv2.insert("", "end", values=row)

    csvTable2.grid(sticky='n',ipadx=400,ipady=250, pady=25,column=1,row=3)

    return OutputDataframe

class GraphPage(ttk.Frame):
    def __init__(self, container, start_page, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        plot_frame = Frame(container)
        plot_frame.pack(fill=BOTH, expand=1)

        my_canvas = Canvas(plot_frame)
        my_canvas.pack(side=LEFT, fill=BOTH, expand=1)

        my_canvas.bind_all("<MouseWheel>",lambda event: my_canvas.yview_scroll(int(-1*(event.delta/120)), "units"))

        my_scrollbar = ttk.Scrollbar(plot_frame, orient=VERTICAL, command=my_canvas.yview)
        my_scrollbar.pack(side=RIGHT, fill=Y)

        my_canvas.configure(yscrollcommand=my_scrollbar.set)
    
        my_canvas.bind(
                "<Configure>",
                lambda e: my_canvas.configure(
                    scrollregion=my_canvas.bbox("all")
                )
            )

        second_frame = Frame(my_canvas, bg=backgroundcolor)

        my_canvas.create_window((0,0), window=second_frame, anchor="nw")

        f = Figure(figsize=(10,10), dpi=100)
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
                            command=lambda: [f.set_figheight(10),f.set_figwidth(20),f.savefig('downloads/duingroeivoorspelling.png')])
        downloadgraph.grid(sticky="e",row=1,column=1)

        downloadcsv = Button(second_frame,state = NORMAL, text="Download csv", width=18,
                            command=lambda: [OutputDataframe.to_csv('downloads/PredictedOutputs.csv', index=False)])
        downloadcsv.grid(sticky="w",row=1,column=1)
            
def main():
    root = tk.Tk()
    root.geometry('1400x800')

    Mainscreen(root)
    
    root.minsize(900,700)
    root.mainloop()
    
main()

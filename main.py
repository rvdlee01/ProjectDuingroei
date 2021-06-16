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
backgroundcolour="floral white"
labelfont = ('calibre',10, 'bold')
xcolumnnames = ['windkracht6','windkracht7','windkracht8','windkracht9','windkracht10','windkracht11','windkracht12','noord','oost','zuid','west','noord-oosten','zuid-oosten','zuid-westen','noord-westen','hogeluchtvochtigheid',
                'lageluchtvochtigheid','gemiddeldeluchtvochtigheid','neerslag']
ycolumnname = 'duinhoogte'
hovercolour = "LightBlue2"
buttoncolour = "snow"

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
    df = df.sort_values(by=['jaar'])

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
    try:
        datasetPredict = model.predict(x_dataset)
        inputPredict = model.predict(x_userinput)
    except:
        exec(open('model.py').read())
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

def on_enter(e, colour):
    e.widget['background'] = colour

def on_leave(e, colour):
    e.widget['background'] = colour

class Mainscreen(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        main_frame = Frame(container)
        main_frame.pack(fill=BOTH, expand=1)

        my_canvas = Canvas(main_frame, bg=backgroundcolour)
        my_canvas.pack(side=LEFT, fill=BOTH, expand=1)

        second_frame = Frame(my_canvas, bg=backgroundcolour)

        frame_n_canvas_id = my_canvas.create_window((0,0), window=second_frame, anchor = "nw")

        my_canvas.bind_all("<MouseWheel>",lambda event: my_canvas.yview_scroll(int(-1*(event.delta/120)), "units"))

        my_scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=my_canvas.yview)
        my_scrollbar.pack(side=RIGHT, fill=Y)

        my_canvas.configure(yscrollcommand=my_scrollbar.set)

        main_frame.bind(
                "<Configure>",
                lambda e: my_canvas.configure(
                    scrollregion=my_canvas.bbox("all")
                )
            )
        my_canvas.bind("<Configure>",
                       lambda event: event.widget.itemconfigure(
                           frame_n_canvas_id, width=event.widget.winfo_width()
                           )
                       )

        predictbutton = Button(second_frame, text="Voorspellen", width=18, bg=buttoncolour,
                            command=lambda: checkInputs())
        predictbutton.grid(row=0,column=2,padx=10,pady=15,ipady=5)
        predictbutton.bind("<Enter>", lambda e: on_enter(e, hovercolour))
        predictbutton.bind("<Leave>", lambda e: on_leave(e, buttoncolour))
        uploadbutton = Button(second_frame, text="CSV bestand selecteren", width=18, bg=buttoncolour,
                            command=lambda: getCsvFile(uploadbutton, predictbutton, tv1))
        uploadbutton.grid(row=0,column=1,padx=10,pady=15,ipady=5)
        uploadbutton.bind("<Enter>", lambda e: on_enter(e, hovercolour))
        uploadbutton.bind("<Leave>", lambda e: on_leave(e, buttoncolour))

        #Close help page window
        def on_closing():
            self.helppage.destroy()
            self.helppageactived = False

        #Create help page window
        def helppage(root):
            if self.helppageactived == False:
                helpWindow = Toplevel(root, bg=backgroundcolour)
                helpWindow.title("Hulppagina")
                helpWindow.geometry("700x600")

                #create scrollable frame for new window
                help_main_frame = Frame(helpWindow)
                help_main_frame.pack(fill=BOTH, expand=1)

                helpCanvas = Canvas(help_main_frame, bg=backgroundcolour)
                helpCanvas.pack(side=LEFT, fill=BOTH, expand=1)

                help_second_frame = Frame(helpCanvas, bg=backgroundcolour)

                help_frame_n_canvas_id = helpCanvas.create_window((0,0), window=help_second_frame, anchor = "nw")

                helpCanvas.bind("<MouseWheel>",lambda event: helpCanvas.yview_scroll(int(-1*(event.delta/120)), "units"))

                helpScrollbar = ttk.Scrollbar(help_main_frame, orient=VERTICAL, command=helpCanvas.yview)
                helpScrollbar.pack(side=RIGHT, fill=Y)

                helpCanvas.configure(yscrollcommand=helpScrollbar.set)

                helpWindow.bind(
                        "<Configure>",
                        lambda e: helpCanvas.configure(
                            scrollregion=helpCanvas.bbox("all")
                        )
                    )

                helpCanvas.bind("<Configure>",
                       lambda event: event.widget.itemconfigure(
                           help_frame_n_canvas_id, width=event.widget.winfo_width()
                           )
                       )
                
                with open('helppage.txt') as f:
                    contents = f.read()
                Label(help_second_frame,text = contents,anchor='nw',justify=LEFT,bg="white",font=('calibre',10)).pack()
                self.helppageactived = True
                self.helppage = helpWindow
                helpWindow.protocol("WM_DELETE_WINDOW", on_closing)
            else:
                #flash window
                self.helppage.focus_force()
                #activate bell sound if the help page is already opened
                self.helppage.bell()

        self.helppageactived = False
        self.helppage = None
        helpbutton = Button(second_frame, text ="Hulp nodig?", bg=buttoncolour, width=18, command = lambda: [helppage(container)])
        helpbutton.grid(row=0,column=3,padx=10,pady=15,ipady=5)
        helpbutton.bind("<Enter>", lambda e: on_enter(e, hovercolour))
        helpbutton.bind("<Leave>", lambda e: on_leave(e, buttoncolour))
        
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
                    messagebox.showerror("Foutmelding", "Vul een geldig numerieke waarde in")
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
                messagebox.showerror("Foutmelding", message)

                return False
            
        vcmd = self.register(validateInput)

        rownumber = 2
        
        for value in listOfInputVariables:
            # Variables for storing data to predict dune height
            setattr(self,value,StringVar())
            if value == 'year':
                # Year entry box and label
                setattr(self,value+'_label',Label(second_frame, text = 'Jaar', font=labelfont, bg=backgroundcolour).grid(padx=30,row=rownumber,column=0,sticky="e"))
            elif 'wp' in value:
                # Wind power entry boxes and labels
                setattr(self,value+'_label',Label(second_frame, text = 'Aantal dagen met windkracht ' + str(count), font=labelfont, bg=backgroundcolour).grid(padx=30,row=rownumber,column=0,sticky="e"))
                count += 1
                columnnumber = 1
            elif value in dictOfDirections.keys():
                # Wind direction entry boxes and labels
                for k, v in dictOfDirections.items():
                    if k == value:
                        textInputWD = 'Aantal dagen met wind vanuit het ' + v
                setattr(self,value+'_label',Label(second_frame, text = textInputWD, font=labelfont, bg=backgroundcolour).grid(padx=30,row=rownumber,column=2,sticky="e"))
                columnnumber = 3
            elif value in dictOfHumidity.keys():
                # Humidity entry boxes and labels
                for k, v in dictOfHumidity.items():
                    if k == value:
                        textInputH = 'Aantal dagen met een ' + v
                setattr(self,value+'_label',Label(second_frame, text = textInputH, font=labelfont, bg=backgroundcolour).grid(padx=30,row=rownumber,column=4,sticky="e"))
                columnnumber = 5
            elif value == 'precipitation':
                # Precipitation entry box and label
                setattr(self,value+'_label',Label(second_frame, text = 'Neerslag in een jaar', font=labelfont, bg=backgroundcolour).grid(padx=30,row=rownumber,column=4,sticky="e"))
                columnnumber = 5
            if value != 'year':
                inputfield = Entry(second_frame, textvariable = getattr(self,value),validate='key', validatecommand=(vcmd,value,'%P'), state=DISABLED)
                inputfield.grid(row=rownumber,column=columnnumber,ipadx=5,pady=5,ipady=5,sticky="ew")
                setattr(self,'entry'+value,inputfield)
            else:
                setattr(self,value,StringVar(second_frame))
                getattr(self,value).set("Selecteer een jaar") # default value
                options = ()
                selectbox = OptionMenu(second_frame, getattr(self,value), options)
                selectbox.config(bg = buttoncolour, activebackground = hovercolour)
                selectbox["menu"].config(bg = buttoncolour)
                selectbox.grid(ipadx=5,ipady=5,row=rownumber, column=1,sticky="ew")
                selectbox['state'] = DISABLED
                setattr(self,'entry'+value,selectbox)
            rownumber += 1
            if value == 'wp12' or value == 'northwest':
                rownumber = 2
        
        # Frame for Treeview
        csvTable = LabelFrame(second_frame,text="CSV data", bg=backgroundcolour)
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
            boolSelectedYear = True
            for inputfield in listOfInputVariables:
                if(inputfield != 'year'):
                    if(len(getattr(self, 'entry'+inputfield).get()) == 0):
                        boolInputs = False
                else:
                    if(len(getattr(self,inputfield).get()) != 4):
                        boolSelectedYear = False
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
            if((boolInputs and boolSelectedYear) and (sumOfWP == sumOfDirections and sumOfWP == sumOfHumidity)):
                my_canvas.pack_forget(), GraphPage(container, self),my_scrollbar.pack_forget(),main_frame.pack_forget()
            else:
                if(filename == ''):
                    messagebox.showerror("Foutmelding", "Selecteer eerst een CSV bestand en vul vervolgens de onderstaande velden in!")
                elif((boolInputs == False) or (boolSelectedYear == False)):
                    if((boolInputs == False) and (boolSelectedYear != False)):
                        messagebox.showerror("Foutmelding", "Vul eerst alle velden in!")
                    elif((boolInputs != False) and (boolSelectedYear == False)):
                        messagebox.showerror("Foutmelding", "Selecteer eerst een jaar voor de voorspelling!")
                    else:
                        messagebox.showerror("Foutmelding", "Vul eerst alle velden in en selecteer een jaar voor de voorspelling!")
                elif(sumOfWP != sumOfDirections or sumOfWP != sumOfHumidity):
                    messagebox.showerror("Foutmelding", "De totaal aantal dagen komen niet met elkaar overeen!\nTotaal aantal dagen bij windkrachten: " + str(sumOfWP)
                                         + "\nTotaal aantal dagen bij windrichtingen: " + str(sumOfDirections) + "\nTotaal aantal dagen bij luchtvochtigheid: " + str(sumOfHumidity))

        def checkColumnNames(checklist, csv_columns):
            for element in checklist:
                if element not in csv_columns:
                    return False
            return True

        def checkRows(selected_file):
            num_rows = -1
            for row in open(selected_file):
                num_rows += 1
            if(num_rows > 10):
                return True
            else:
                return False

        def checkValues(df):
            hasEmptyValues = df.isnull().values.any()
            if(hasEmptyValues):
                return False
            else:
                return True

        def checkYear(df):
            df = df.sort_values(by=['jaar'])
            start = df['jaar'].iloc[0]
            end = df['jaar'].iloc[-1]
            missingyears = ""
            for year in range(start,end+1,1):
                if year not in list(df.jaar):
                    missingyears += " {}".format(year)
            if missingyears == "":
                return True,missingyears
            else:
                missingyears = "Het bestand bevat de volgende ontbrekende jaren:" + missingyears
                return False,missingyears
                    
        def load_Data(tv1,csv_data):
            try:
                df = pd.read_csv(csv_data)
                
                if 'jaar' in df:
                    boolYear,missingyears = checkYear(df)
                    if (boolYear):
                        df = df.sort_values(by=['jaar'])
                else:
                    boolYear,missingyears = False, "Het bestand bevat geen kolom 'jaar'"  
                      
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
                if 'jaar' in df:
                    lastyear = df['jaar'].iloc[-1]
                else:
                    lastyear = ""
                #detect outliers
                #outliers = detect_outlier(df['duinhoogte'])
                #print('outliers: ', outliers)
                return boolColumns, boolRows, boolValues, boolYear, missingyears, lastyear
            except:
                uploadbutton.configure(bg="red", activebackground=hovercolour)
                uploadbutton.bind("<Leave>", lambda e: on_enter(e, "red"))
                tk.messagebox.showerror("Error", "Ongeldig bestand")
            return None

        def getCsvFile(uploadbutton, predictbutton, tv1): #treeview, buttons
            global filename
            filename = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = [("CSV files", '.csv')])
            if filename != '':
                boolColumns, boolRows, boolValues, boolYear, missingyears, lastyear = load_Data(tv1,filename)
                if ((boolColumns == False) or (boolRows == False) or (boolValues == False) or (boolYear == False)):
                    for inputfield in listOfInputVariables:
                        getattr(self, 'entry'+inputfield)["state"] = DISABLED
                    uploadbutton.configure(bg="red", activebackground=hovercolour)
                    uploadbutton.bind("<Leave>", lambda e: on_enter(e, "red"))
                    csvErrorMessage = ""
                    if(boolColumns == False):
                        csvErrorMessage += "Kolomnamen komen niet overeen met de vereiste kolomnamen!\n"
                    if(boolRows == False):
                        csvErrorMessage += "Te weinig rijen!\n"
                    if(boolValues == False):
                        csvErrorMessage += "Er ontbreken gegevens!\n"
                    if(boolYear == False):
                        csvErrorMessage += missingyears
                    messagebox.showerror("Foutmelding", csvErrorMessage)
                if ((boolColumns == True) and (boolRows == True) and (boolValues == True) and (boolYear == True)):
                    for inputfield in listOfInputVariables:
                        getattr(self, 'entry'+inputfield)["state"] = NORMAL
                    uploadbutton.configure(bg="lime green", activebackground=hovercolour)
                    uploadbutton.bind("<Leave>", lambda e: on_enter(e, "lime green"))
                    selectbox['menu'].delete(0,'end')
                    options = (lastyear + 1)
                    selectbox['menu'].add_command(label=options, command=tk._setit(getattr(self,'year'), options))
                    messagebox.showinfo("Bevestiging", "CSV bestand succesvol geupload")

def clearGraphpage(canvas):
    for item in canvas.get_tk_widget().find_all():
        canvas.get_tk_widget().delete(item)

def hideGraph(canvas,graph,histogram, showbarplot):
    if(graph.get_visible()):
        graph.set_visible(False)
        histogram.set_visible(True)
        showbarplot["text"]="Laat grafiek zien"
    else:
        histogram.set_visible(False)
        graph.set_visible(True)
        showbarplot["text"]="Laat histogram zien"
    canvas.draw()

def plotGraph(a,b,f,canvas,startpage,tv2,csvTable2):
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
    a.plot(datasetPredictedx, datasetPredictedy,label='Voorspelde waarde',color='lightsalmon')
    a.plot(datasetActualx, datasetActualy,label='Daadwerkelijke waarde',color='turquoise')
    a.scatter(userOutputx,userOutputy,label='Voorspelde invoergegevens van gebruiker',color='royalblue')

    a.legend()
    a.set_xlabel('jaar',labelpad=10)
    a.set_ylabel('duinhoogte [cm]',labelpad=10)
    
    actualY = []
    predictedX = []

    for i in datasetActualy:
        actualY.append(i[0])
    for i in datasetPredictedy:
        predictedX.append(i[0])

    barWidth = 0.4
    
    # set heights of bars
    actualY.append(0) # add extra value for 2021 not visible
    predictedX.append(0)

    bars1 = actualY
    bars2 = predictedX

    user_input_list = []
    for i in datasetActualx:
        user_input_list.append(0)
    user_input_list.append(userOutputy[0][0])

    bars3 = user_input_list
    
    # Set position of bar on X axis
    r1 = OutputDataframe["Jaar"]
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    b.bar(r1, bars1, color='turquoise', width=barWidth, edgecolor='white', label='Daadwerkelijke waarde')
    b.bar(r2, bars2, color='lightsalmon', width=barWidth, edgecolor='white', label='Voorspelde waarde')
    b.bar(r3, bars3, color='royalblue', width=barWidth, edgecolor='white', label='Voorspelde invoergegevens van gebruiker')
    
    # Create legend
    b.set_xlabel('jaar',labelpad=10)
    b.set_ylabel('duinhoogte [cm]',labelpad=10)
    b.legend()
    b.set_visible(False)

    canvas.draw()
    canvas.get_tk_widget().grid(pady=25,row=2,columnspan=3)

    for i in tv2.get_children():
        tv2.delete(i)

    tv2["column"] = list(OutputDataframe.columns)
    tv2["show"] = "headings"
    for column in tv2["columns"]:
        tv2.heading(column, text=column)
            
    df_rows = OutputDataframe.to_numpy().tolist()
    for row in df_rows:
        tv2.insert("", "end", values=row)

    csvTable2.grid(ipadx=400,ipady=250, pady=25,row=3,columnspan=3)

    return OutputDataframe

class GraphPage(ttk.Frame):
    def __init__(self, container, start_page, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        plot_frame = Frame(container)
        plot_frame.pack(fill=BOTH, expand=1)

        my_canvas = Canvas(plot_frame, bg=backgroundcolour)
        my_canvas.pack(side=LEFT, fill=BOTH, expand=1)

        second_frame = Frame(my_canvas, bg=backgroundcolour)

        frame_n_canvas_id = my_canvas.create_window((0,0), window=second_frame, anchor="nw")

        my_canvas.bind_all("<MouseWheel>",lambda event: my_canvas.yview_scroll(int(-1*(event.delta/120)), "units"))

        my_scrollbar = ttk.Scrollbar(plot_frame, orient=VERTICAL, command=my_canvas.yview)
        my_scrollbar.pack(side=RIGHT, fill=Y)

        my_canvas.configure(yscrollcommand=my_scrollbar.set)
    
        plot_frame.bind(
                "<Configure>",
                lambda e: my_canvas.configure(
                    scrollregion=my_canvas.bbox("all")
                )
            )
        
        my_canvas.bind("<Configure>",
                       lambda event: event.widget.itemconfigure(
                           frame_n_canvas_id, width=event.widget.winfo_width()
                           )
                       )

        f = Figure(figsize=(10,5), dpi=100)
        a = f.add_subplot(111)
        b = f.add_subplot(111)
        canvas = FigureCanvasTkAgg(f, second_frame)

        # Frame for Treeview
        csvTable2 = LabelFrame(second_frame,text="Grafiek gegevens",bg=backgroundcolour)

        # Treeview Widget
        tv2 = ttk.Treeview(csvTable2)
        tv2.place(relheight=1, relwidth=1)

        treescrolly = Scrollbar(csvTable2, orient="vertical", command=tv2.yview)
        treescrollx = Scrollbar(csvTable2, orient="horizontal", command=tv2.xview)
        tv2.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set)
        treescrollx.pack(side="bottom", fill="x")
        treescrolly.pack(side="right", fill="y")

        OutputDataframe = plotGraph(a,b,f,canvas,start_page,tv2,csvTable2)

        homebutton = Button(second_frame,state = NORMAL, text="Terug naar startpagina", width=18, bg=buttoncolour,
                            command=lambda: [my_canvas.pack_forget(),csvTable2.grid_forget(),canvas.get_tk_widget().pack_forget(),clearGraphpage(canvas),my_scrollbar.pack_forget(),plot_frame.pack_forget(),Mainscreen(container)])
        homebutton.grid(padx=10,pady=15,ipady=5,row=0,column=1)
        
        homebutton.bind("<Enter>", lambda e: on_enter(e, hovercolour))
        homebutton.bind("<Leave>", lambda e: on_leave(e, buttoncolour))

        downloadgraph = Button(second_frame,state = NORMAL, text="Download grafiek", width=18, bg=buttoncolour,
                            command=lambda: [f.savefig('downloads/duingroeivoorspelling.png', dpi=100), messagebox.showinfo("Download successvol", "Bestand opgeslagen naar: downloads/duingroeivoorspelling.png in de application folder")])
        downloadgraph.grid(padx=10,ipady=5,row=1,column=0)
        
        downloadgraph.bind("<Enter>", lambda e: on_enter(e, hovercolour))
        downloadgraph.bind("<Leave>", lambda e: on_leave(e, buttoncolour))

        downloadcsv = Button(second_frame,state = NORMAL, text="Download CSV", width=18, bg=buttoncolour,
                            command=lambda: [OutputDataframe.to_csv('downloads/PredictedOutputs.csv', index=False), messagebox.showinfo("Download successvol", "Bestand opgeslagen naar: downloads/PredictedOutputs.csv in de applicatie folder")])
        downloadcsv.grid(padx=10,ipady=5,row=1,column=2)
        
        downloadcsv.bind("<Enter>", lambda e: on_enter(e, hovercolour))
        downloadcsv.bind("<Leave>", lambda e: on_leave(e, buttoncolour))

        showbarplot = Button(second_frame,state = NORMAL, text="Laat histogram zien", width=18, bg=buttoncolour,
                            command=lambda: [hideGraph(canvas,a,b, showbarplot)])
        showbarplot.grid(padx=10,ipady=5,row=1,column=3)

        showbarplot.bind("<Enter>", lambda e: on_enter(e, hovercolour))
        showbarplot.bind("<Leave>", lambda e: on_leave(e, buttoncolour))

        def getAccuracy(OutputDataframe):
            acc = []
            for errorrate in OutputDataframe['Foutpercentage'][:len(OutputDataframe['Foutpercentage']) - 1]:
                acc.append(1-(abs(errorrate)/100))
            accuracy = sum(acc)/len(acc)
            return round(accuracy*100,2)
    
        #label accuracy
        accuracyLabel = Label(second_frame, text="Nauwkeurigheid van model: "+str(getAccuracy(OutputDataframe))+"%", bg=backgroundcolour )
        accuracyLabel.grid(padx=10,pady=5,row=1,column=1)
            
def main():
    root = tk.Tk()
    root.geometry('1500x800')

    Mainscreen(root)
    
    root.minsize(900,700)
    root.mainloop()
    
main()

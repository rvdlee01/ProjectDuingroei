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
    except:
        tk.messagebox.showerror("Error", "Wrong file or file format!")
    return None



def getCsvFile(self):
    global filename
    filename = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("Csv files", "*.csv"),("xlsx files","*.xlsx")))
    if filename != '':
        self.button1["state"] = tk.NORMAL
        self.button2.configure(bg="green")
        self.tv1.delete(*self.tv1.get_children()) #Empties rows and columns in csvTable
        load_Data(self,filename)
        


class DuinGroeiApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.iconbitmap(self, default="dune_icon.ico")
        tk.Tk.wm_title(self, "DuinHoogte")
        tk.Tk.wm_geometry(self, "1600x900")
        

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
                            command=lambda: [controller.show_frame(PageOne),self.csvTable.place_forget()], font = myFont)
        self.button2 = tk.Button(self, text="Select csv file",
                            command=lambda: getCsvFile(self), font = myFont)
        self.button1.pack(padx=50,pady=50)
        self.button2.pack()
        
        # Frame for Treeview
        self.csvTable = tk.LabelFrame(text ="CSV data")
        self.csvTable.place(height=450,width=800,x = 400, y = 350)


        ## Treeview Widget
        self.tv1 = ttk.Treeview(self.csvTable)
        self.tv1.place(relheight=1, relwidth=1)

        treescrolly = tk.Scrollbar(self.csvTable, orient= "vertical", command = self.tv1.yview)
        treescrollx = tk.Scrollbar(self.csvTable, orient= "horizontal", command = self.tv1.xview)
        self.tv1.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set)
        treescrollx.pack(side="bottom", fill="x")
        treescrolly.pack(side="right", fill="y")

        # self.csvTable.destroy()


class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        start_page = self.controller.get_page(StartPage) # Access variables from another class
        label = tk.Label(self, text="Graph page", font=LARGE_FONT)
        label.pack(pady=10,padx=10)
        button1 = ttk.Button(self, text="Home",
                            command=lambda: [controller.show_frame(StartPage), start_page.csvTable.place(height=450,width=800,x = 400, y = 350)])
        button1.pack()

        f = Figure(figsize=(5,5), dpi=100)
        a = f.add_subplot(111)
        a.plot([1,2,3,4,5,6,7,8],[5,6,7,3,4,5,6,9])

        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand= True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

app = DuinGroeiApp()
app.mainloop()
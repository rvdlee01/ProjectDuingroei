from tkinter.constants import DISABLED
import matplotlib
matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from tkinter import *
import tkinter as tk
from tkinter import ttk

from tkinter import font

LARGE_FONT = ("Verdana", 12)
filename =''
def getCsvFile(self):
    global filename
    filename = filedialog.askopenfilename()
    if filename != '':
        self.button1["state"] = tk.NORMAL
        self.button2.configure(bg="green")
    

class DuinGroeiApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.iconbitmap(self, default="dune_icon.ico")
        tk.Tk.wm_title(self, "DuinHoogte")

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

class StartPage(tk.Frame):
    global button1
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        myFont = font.Font(family = 'Helvetica', size = 30)
        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        self.button1 = tk.Button(self,state = DISABLED, text="Go to graph page",
                            command=lambda: controller.show_frame(PageOne), font = myFont)
        self.button2 = tk.Button(self, text="Select csv file",
                            command=lambda: getCsvFile(self), font = myFont)
        self.button1.pack(padx=50,pady=50)
        self.button2.pack()

class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Graph page", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        f = Figure(figsize=(5,5), dpi=100)
        a = f.add_subplot(111)
        a.plot([1,2,3,4,5,6,7,8],[5,6,7,3,4,5,6,9])

        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand= True)

        

app = DuinGroeiApp()
app.mainloop()
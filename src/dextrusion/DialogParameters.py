
import tkinter as tk
from tkinter import filedialog
import os

class DialogParameters:
    
    def finish(self):
        self.bs = self.a.get()
        self.epochs = self.ep.get()
        self.scaling = self.sc.get()
        self.aug = self.ag.get()
        self.ratio = self.val.get()
        self.window.destroy()
    
    def askmodeldirectory(self):
        self.modelf = filedialog.askdirectory(initialdir=os.getcwd(), title="Go inside model to retrain directory")
    
    def askdatadirectory(self):
        self.path = filedialog.askdirectory(initialdir=os.getcwd(), title="Go inside retraining data directory")

    def dialog_retrain(self):    
        self.window = tk.Tk()
        self.window.title("GUI")
        
        L4 = tk.Label(self.window, text="Model to retrain:")
        L4.grid(row=0, column=0)
        self.modelf = os.getcwd()
        mfile = tk.Button(self.window, text='Browse', command=self.askmodeldirectory)
        mfile.grid(row=0, column=1)
        
        L6 = tk.Label(self.window, text="Retrain dataset path:")
        L6.grid(row=0, column=2)
        self.path = os.getcwd()
        dfile = tk.Button(self.window, text='Browse', command=self.askdatadirectory)
        dfile.grid(row=0, column=3)

        L1 = tk.Label(self.window, text="Batch size:")
        self.a = tk.Entry(self.window)
        self.a.insert(0,"50")
        L1.grid(row=1, column=0)
        self.a.grid(row=1, column=1)

        
        L2 = tk.Label(self.window, text="Nb epochs to retrain:")
        self.ep = tk.Entry(self.window)
        self.ep.insert(0,"10")
        L2.grid(row=1, column=2)
        self.ep.grid(row=1, column=3)
        
        L3 = tk.Label(self.window, text="Images scaling (in µm):")
        self.sc = tk.Entry(self.window)
        self.sc.insert(0,"0.275")
        L3.grid(row=2, column=0)
        self.sc.grid(row=2, column=1)
        
        L5 = tk.Label(self.window, text="Augment data (1 no aug., >1 aug.):")
        self.ag = tk.Entry(self.window)
        self.ag.insert(0,"3")
        L5.grid(row=2, column=2)
        self.ag.grid(row=2, column=3)
        
        L7 = tk.Label(self.window, text="Train/Validation split ratio:")
        self.val = tk.Entry(self.window)
        self.val.insert(0,"0.2")
        L7.grid(row=3, column=0)
        self.val.grid(row=3, column=1)


        
        but = tk.Button(self.window, text = 'OK', command = self.finish)#.grid(column = 1, row = 3)
        but.grid(row=3, column=2)
        self.window.mainloop()

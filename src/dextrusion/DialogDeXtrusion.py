import tkinter as tk
from tkinter import filedialog, ttk
import os, re
from glob import glob

class DialogDeXtrusion:
 

    def finish(self):
        self.cell_diameter = self.celld.get()
        self.extrusion_duration = self.extd.get()
        self.dxyval = self.dxy.get()
        self.dtval = self.dt.get()
        self.modeldir = self.dirname.get()
        self.imagepath = self.imname.get()
        self.imagepath = self.tolist(self.imagepath)
        self.saveProba = (self.varprob.get() == 1)
        self.saveProbaOne = (self.varprobone.get() == 1)
        self.saveRois = (self.varroi.get() == 1)
        self.proba_vol = self.pvol.get()
        self.threshold = self.pthres.get()
        self.event = self.list.get()
        self.event_ind = self.list.current()
        self.group_size = self.grsize.get()
        self.window.destroy()
    

    def tolist(self, imagepath):
        res = []
        imagepath = imagepath.replace("(","")
        imagepath = imagepath.replace(")","")
        imagepath = imagepath.replace("'","")
        imagepath = imagepath.replace("\"","")
        imagepath = imagepath.replace("[","")
        imagepath = imagepath.replace("]","")
        images = re.split(',', imagepath)
        #print(images)
        for im in images:
            im = im.lstrip(" ")
            res.append(im)
        return res


    def askmodeldirectory(self):
        self.model = filedialog.askdirectory(initialdir=self.model, title="Go inside model directory")
        self.dirname.delete(0, tk.END)
        self.dirname.insert(0,str(self.model))
        self.catlist = self.read_catnames_fromconfiguration(self.model)
    
    def read_catnames_fromconfiguration(self, modelpath):
        """ Read configuration file that is automatically saved in the model folder 'modelpath' to set the parameters """
        configpath = os.path.join(modelpath,'config.cfg')
        if not os.path.exists(configpath):
            path = glob(modelpath+"/*/", recursive=False)
            print(path)
            if len(path) <=0:
                self.list['values'] = ["", "_cell_death.zip"]
                return
            i = 0
            while not os.path.exists(configpath):
                print(path[i])
                configpath = os.path.join(path[i],'config.cfg')
                i = i + 1
                if i > len(path):
                    self.list['values'] =  ["", "_cell_death.zip"]
                    return

        configfile = open(configpath, 'r')
        lines = configfile.readlines()
        for line in lines:
            vals = line.split("=")
            if len(vals) >= 1:
                if vals[0].strip() == "events_category_names":
                    self.list['values'] = self.read_catnames(vals[1])
                    return
        self.list['values'] = ["", "_cell_death.zip"]
    
    def read_catnames(self, instr):
        catnames = []
        events = re.split(',|\[|\]|\n', instr)
        for ind in range(1,len(events)-2):
            catnames.append(events[ind].strip().replace("'",""))
        return catnames

        
        ## Bounds: size of area to fill with probability of the window calculated (not all the window, around the center)
        self.bounds = ( floor(self.nframes[0]*0.2), floor(self.nframes[1]*0.2)+1, floor(self.half_size[0]*0.3), floor(self.half_size[1]*0.3) )
        
    def askmovie(self):
        self.impath = filedialog.askopenfilenames()
        self.imname.delete(0, tk.END)
        self.imname.insert(0,str(self.impath))
    
    def dialog_main(self, model_dir=os.getcwd(), impath=os.getcwd()):  
        self.window = tk.Tk()
        self.window.title("DeXtrusion")
        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_columnconfigure(1, weight=1)
        self.window.grid_columnconfigure(2, weight=8)
        self.model = model_dir
        self.impath = impath
        self.list = ttk.Combobox(self.window, values = [])
        self.list.set("Choose event to detect:")

        lastrow = 0
        pathw = 50
        numw = 10
        lastrow = lastrow + 1
        Ltrt = tk.Label(self.window, text="Choose network to use, go inside the model folder for that. If you want to run several network, they all should be in that folder.")
        Ltrt.grid(row=lastrow, column=0, columnspan=4)
        lastrow = lastrow + 1
        
        LM = tk.Label(self.window, text="Trained model(s) to use :")
        LM.grid(row=lastrow, column=0)
        mfile = tk.Button(self.window, text='Browse', command=self.askmodeldirectory)
        mfile.grid(row=lastrow, column=1)
        self.dirname = tk.Entry(self.window, width=pathw)
        self.dirname.grid(row=lastrow, column=2, columnspan=3)
        self.dirname.delete(0,tk.END)
        self.dirname.insert(0, str(self.model))
        self.read_catnames_fromconfiguration(self.model)
        
        lastrow = lastrow + 1
        Lempty = tk.Label(self.window, text="")
        Lempty.grid(row=lastrow, column=0)
        
        lastrow = lastrow + 1
        Ltmt = tk.Label(self.window, text="Choose on which movie(s) to process. You can select several movies.")
        Ltmt.grid(row=lastrow, column=0, columnspan=4)
        
        lastrow = lastrow + 1
        LIm = tk.Label(self.window, text="Input movie(s) :")
        LIm.grid(row=lastrow, column=0)
        img = tk.Button(self.window, text='Choose movie(s):', command=self.askmovie)
        img.grid(row=lastrow, column=1)
        self.imname = tk.Entry(self.window, width=pathw)
        self.imname.grid(row=lastrow, column=2, columnspan=3)
        self.imname.delete(0,tk.END)
        self.imname.insert(0, str(self.impath))
        
        lastrow = lastrow + 1
        empty = tk.Label(self.window, text="")
        empty.grid(row=lastrow, column=0)
        
        lastrow = lastrow + 1
        Lpar = tk.Label(self.window, text="Movie resolution:")
        Lpar.grid(row=lastrow, column=0, columnspan=2)
        
        Lpart = tk.Label(self.window, text="Windows shift:")
        Lpart.grid(row=lastrow, column=3, columnspan=2)
        
        
        lastrow = lastrow + 1
        Lc = tk.Label(self.window, text="Typical cell diameter (pixels):")
        self.celld = tk.Entry(self.window, width=numw)
        self.celld.insert(0,"25.0")
        Lc.grid(row=lastrow, column=0)
        self.celld.grid(row=lastrow, column=1)
        
        empty2 = tk.Label(self.window, text="", width=10)
        empty2.grid(row=lastrow, column=2)
        
        Lxy= tk.Label(self.window, text="spatial shift:")
        self.dxy= tk.Entry(self.window, width=numw)
        self.dxy.insert(0,"10")
        Lxy.grid(row=lastrow, column=3)
        self.dxy.grid(row=lastrow, column=4)
        
        lastrow = lastrow + 1
        Ld= tk.Label(self.window, text="Typical extrusion duration (frames):")
        self.extd = tk.Entry(self.window, width=numw)
        self.extd.insert(0,"4.5")
        Ld.grid(row=lastrow, column=0)
        self.extd.grid(row=lastrow, column=1)

        Lt= tk.Label(self.window, text="Temporal shift:")
        self.dt= tk.Entry(self.window, width=numw)
        self.dt.insert(0,"2")
        Lt.grid(row=lastrow, column=3)
        self.dt.grid(row=lastrow, column=4)
        
        lastrow = lastrow + 1
        empty3 = tk.Label(self.window, text="")
        empty3.grid(row=lastrow, column=0)
        
        lastrow = lastrow + 1
        Lpar = tk.Label(self.window, text="Results output:")
        Lpar.grid(row=lastrow, column=0, columnspan=2)
        
        lastrow = lastrow + 1
        Lpar = tk.Label(self.window, text="Event to ROI:")
        Lpar.grid(row=lastrow, column=0)
        
        self.list.grid(row=lastrow, column=1)
        self.list.set('_cell_death.zip')
        
        lastrow = lastrow + 1
        self.varroi = tk.IntVar(value=1)
        croi = tk.Checkbutton(self.window, text='Fiji ROIs of event \t     ',variable=self.varroi, onvalue=1, offvalue=0)
        croi.grid(row=lastrow, column=0)
        
        Lvol = tk.Label(self.window, text="Min probability volume:")
        self.pvol = tk.Entry(self.window, width=numw)
        self.pvol.insert(0,"800")
        Lvol.grid(row=lastrow, column=1)
        self.pvol.grid(row=lastrow, column=2)
        
        Lthres = tk.Label(self.window, text="Min proba threshold (150-255):")
        self.pthres = tk.Entry(self.window, width=numw)
        self.pthres.insert(0,"180")
        Lthres.grid(row=lastrow, column=3)
        self.pthres.grid(row=lastrow, column=4)
        
        lastrow = lastrow + 1
        self.varprob = tk.IntVar(value=1)
        cprob = tk.Checkbutton(self.window, text='Save all probability maps',variable=self.varprob, onvalue=1, offvalue=0)
        cprob.grid(row=lastrow, column=0, columnspan=2)
        
        self.varprobone = tk.IntVar(value=0)
        cprobone = tk.Checkbutton(self.window, text='Save cleaned selected event proba map',variable=self.varprobone, onvalue=1, offvalue=0)
        cprobone.grid(row=lastrow, column=2, columnspan=2)

        lastrow = lastrow + 1
        Lempy = tk.Label(self.window, text="")
        Lempy.grid(row=lastrow, column=0)
        
        lastrow = lastrow + 1
        Ltgst = tk.Label(self.window, text="Computing efficiency, depends on your computer capacity. The higher the faster, but might exceed memory.")
        Ltgst.grid(row=lastrow, column=0, columnspan=2)
        lastrow = lastrow + 1
        
        Lgr= tk.Label(self.window, text="Computing size:")
        Lgr.grid(row=lastrow, column=0)
        self.grsize= tk.Entry(self.window, width=numw)
        self.grsize.insert(0,"150000")
        self.grsize.grid(row=lastrow, column=1)


        lastrow = lastrow + 1
        but = tk.Button(self.window, text = 'OK', command = self.finish)#.grid(column = 1, row = 3)
        but.grid(row=lastrow, column=5)
        self.window.mainloop()

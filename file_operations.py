# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 16:53:36 2021

@author: malrawi
"""



import tkinter
from tkinter import filedialog
import os

def get_file_name():
    #initiate tinker and hide window 
    main_win = tkinter.Tk() 
    main_win.withdraw()
    
    main_win.overrideredirect(True)
    main_win.geometry('0x0+0+0')
    
    main_win.deiconify()
    main_win.lift()
    main_win.focus_force()
    
    #open file selector 
    main_win.sourceFile = tkinter.filedialog.askopenfilename(parent=main_win, initialdir= "/",
    title='Please select a file')
    
    #close window after selection 
    main_win.destroy()
    
    #print path 
    file_name = os.path.basename(main_win.sourceFile) # print(main_win.sourceFile )
    folder_name = os.path.dirname(main_win.sourceFile)
    return file_name, folder_name
    

import tkinter
from tkinter import *
from tkinter.ttk import *  # Automatically replace some widgets with better versions
from application import Core

if __name__ == "__main__":
    # Frame with no parent makes that frame's master the root
    app = Core()
    app.master.geometry("800x400")
    app.master.title("Mahjong Master")
    app.mainloop()


import os
import sys
sys.path.insert(1, os.getcwd() + "\\ml_reinvented_wheel")
from tkinter import *
from tkinter.ttk import *  # Automatically replace some widgets with better versions
from application import Application
from hand_calculator import HandCalculator
import time


if __name__ == "__main__":
    root = Tk()
    app = Application(root)
    root.geometry("1800x800")
    root.title("Mahjong Master")
    root.mainloop()

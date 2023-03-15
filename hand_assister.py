import tkinter
from tkinter import *
from tkinter.ttk import *  # Automatically replace some widgets with better versions
from hands import MahjongHands
from utilities import *

# Similar to hand_calculator.
# Generates the frames and GUI to hold the hand entry, visualizer and nearest hands calculator


class HandAssister(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)


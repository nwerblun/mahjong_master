import tkinter
from tkinter import *
from tkinter.ttk import *  # Automatically replace some widgets with better versions
from hand_calculator import HandCalculator
from utilities import recursive_destroy


class Application(Frame):
    def __init__(self, root, *args, **kwargs):
        Frame.__init__(self, root, *args, **kwargs)
        self.hand_calc = None
        self.pack(expand=YES, fill=BOTH)
        self.app_control_frame = Frame(self, borderwidth=1, relief=GROOVE)
        self.app_control_frame.pack(side=TOP, fill=X)
        self.calc_button = Button(self.app_control_frame, text="Show Calculator")
        self.calc_button.pack(expand=YES, fill=BOTH)
        self.calc_button.configure(command=self.destroy_hand_calc)
        self.hand_calc_frame = Frame(self)
        self.hand_calc_frame.pack(side=BOTTOM, fill=BOTH, expand=YES)
        self.create_hand_calc(self.hand_calc_frame)

    def create_hand_calc(self, root):
        self.hand_calc = HandCalculator(root)
        self.hand_calc.create_hand_table()

    def destroy_hand_calc(self):
        if self.hand_calc is not None:
            recursive_destroy(self.hand_calc_frame)
            self.hand_calc_frame = None
            self.hand_calc = None


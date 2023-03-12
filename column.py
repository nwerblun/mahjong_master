from tkinter import *
from tkinter.ttk import *
import numpy as np


class Column:
    def __init__(self, parent, header, arr):
        self.hidden = False
        self.parent = parent
        self.header = header
        self.data = arr


class LabelColumn(Column):
    pass


class CheckboxColumn(Column):
    pass

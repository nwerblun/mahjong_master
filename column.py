from tkinter import *
from tkinter.ttk import *
import numpy as np


class Column:
    def __init__(self, parent, header, arr, header_style=None):
        self.hidden = False
        self.parent = parent
        self.header = header
        self.data = arr
        self.column_index = -1
        self.header_style = header_style

    def add_to_parent_grid(self, index):
        return

    def get_max_column_width(self):
        return


class LabelColumn(Column):
    def __init__(self, parent, header, arr, header_style=None):
        super().__init__(parent, header, arr, header_style)
        self.header_label = None
        self.data_labels = []
        self.data_locations = []
        for i in range(len(self.data)):
            arr[i] = str(arr[i])

    def add_to_parent_grid(self, index):
        self.header_label = Label(self.parent, text=self.header, style=self.header_style, borderwidth=1, relief=GROOVE)
        self.header_label.grid(row=0, column=index, sticky=N+E+S+W)
        for row_ind, d in enumerate(self.data):
            lbl = Label(self.parent, text=d, borderwidth=1, relief=GROOVE)
            self.data_labels += [lbl]
            lbl.grid(row=row_ind+1, column=index, sticky=N+E+S+W)
            self.data_locations += [(row_ind+1, index)]

    def get_max_column_width(self):
        max_len = len(self.header)
        for d in self.data:
            max_len = max(max_len, len(d))
        return max_len

    def set_wraptext_width(self, width):
        self.header_label.configure(wraplength=width)
        for dl in self.data_labels:
            dl.configure(wraplength=width)


class CheckboxColumn(Column):
    def __init__(self, parent, header, arr, header_style=None):
        super().__init__(parent, header, arr, header_style)

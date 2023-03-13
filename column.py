from tkinter import *
from tkinter.ttk import *
import numpy as np
from abc import ABC, abstractmethod


class Column(ABC):
    def __init__(self, parent, header, arr, index, header_style=None):
        self.hidden = False
        self.parent = parent
        self.header = header
        self.data = arr
        self.column_index = index
        self.header_style = header_style

    @abstractmethod
    def add_to_parent_grid(self, index):
        return

    @abstractmethod
    def get_max_column_text_width(self):
        return

    @abstractmethod
    def get_num_rows(self):
        pass

    @abstractmethod
    def shift_column(self, index):
        self.column_index = index

    @abstractmethod
    def hide(self):
        pass

    @abstractmethod
    def unhide(self):
        pass


class LabelColumn(Column):
    def __init__(self, parent, header, arr, index, header_style=None):
        super().__init__(parent, header, arr, index, header_style)
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

    def get_max_column_text_width(self):
        max_len = len(self.header) * 6  # Account for headers being larger
        for d in self.data:
            max_len = max(max_len, len(d))
        return max_len

    def set_wraptext_width(self, width):
        self.header_label.configure(wraplength=width)
        for dl in self.data_labels:
            dl.configure(wraplength=width)

    def get_num_rows(self):
        return len(self.data)

    def shift_column(self, index):
        self.column_index = index
        self.header_label.grid(column=index)
        for d in self.data_labels:
            d.grid(column=index)
        for i in range(len(self.data_locations)):
            self.data_locations[i] = (i, index)

    def hide(self):
        self.hidden = True
        self.header_label.grid_remove()
        for d in self.data_labels:
            d.grid_remove()

    def unhide(self):
        self.hidden = False
        self.header_label.grid()
        for d in self.data_labels:
            d.grid()


class CheckboxColumn(Column):
    def __init__(self, parent, header, arr, index, header_style=None):
        # Arr should be bools
        super().__init__(parent, header, arr, index, header_style)
        self.header_label = None
        self.checkboxes = {}
        self.frame_locations = []
        self.box_frames = []

    def _toggle_state(self, btn):
        row = self.checkboxes[btn]
        self.data[row] = not self.data[row]

    def add_to_parent_grid(self, index):
        self.header_label = Label(self.parent, text=self.header, style=self.header_style, borderwidth=1, relief=GROOVE)
        self.header_label.grid(row=0, column=index, sticky=N+E+S+W)
        for row_ind, d in enumerate(self.data):
            frm = Frame(self.parent, borderwidth=1, relief=GROOVE)
            frm.grid(row=row_ind+1, column=index, sticky=N+E+S+W)
            self.box_frames += [frm]
            self.frame_locations += [(row_ind+1, index)]
            btn = Checkbutton(frm)
            btn.grid(row=0, column=0, sticky=N+E+S+W)
            self.checkboxes[btn] = row_ind  # No +1 because this will be used to access self.data
            btn.invoke()
            btn.invoke()
            btn.configure(command=lambda b=btn: self._toggle_state(btn))

    def get_max_column_text_width(self):
        return int(len(self.header) * 6)  # Account for larger font size

    def set_wraptext_width(self, width):
        self.header_label.configure(wraplength=width)

    def get_num_rows(self):
        return len(self.data)

    def shift_column(self, index):
        self.column_index = index
        self.header_label.grid(column=index)
        for f in self.box_frames:
            f.grid(column=index)
        for i in range(len(self.frame_locations)):
            self.frame_locations[i] = (i, index)
        for k in self.checkboxes.keys():
            k.grid(column=index)

    def hide(self):
        self.hidden = True
        self.header_label.grid_remove()
        for b in self.box_frames:
            b.grid_remove()
        for k in self.checkboxes.keys():
            k.grid_remove()

    def unhide(self):
        self.hidden = False
        self.header_label.grid()
        for b in self.box_frames:
            b.grid()
        for k in self.checkboxes.keys():
            k.grid()
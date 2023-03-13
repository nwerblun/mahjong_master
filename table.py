from tkinter import *
from tkinter.ttk import *
import numpy as np
from column import *

class TkinterTable:

    def __init__(self, root, table_data):
        # Assumes table data contains the headers as the first row.
        # Must be text data. Non-text data must be added via functions.
        self.root = root
        self.header_font = "Header.TLabel"
        self.num_rows = len(table_data)
        self.columns = []
        header_style = Style()
        header_style.configure("Header.TLabel", font=('Segoe UI', 14, "bold"))
        for i in range(len(table_data[0])):
            self.columns += [LabelColumn(self.root, table_data[0, i],
                                         table_data[1:, i], header_style="Header.TLabel")]

    def get_num_cols(self):
        return len(self.columns)

    def get_num_rows(self):
        return self.num_rows

    def _set_resize_widths(self):
        col_widths = []
        for c in self.columns:
            col_widths += [c.get_max_column_width()]

        total = sum(col_widths)
        for ind, c in enumerate(self.columns):
            self.root.columnconfigure(ind, weight=int((100 * col_widths[ind] / total)))
            c.set_wraptext_width(max(int(self.root.winfo_width() * col_widths[ind] / total), col_widths[ind]))

    def populate(self):
        for ind, c in enumerate(self.columns):
            c.add_to_parent_grid(ind)

    def redraw(self):
        self._set_resize_widths()

    def _configure_rows(self):
        for i in range(self.num_rows):
            self.root.rowconfigure(i, weight=1)

    def _shift_n_columns(self, index, dir=1):
        pass

    def add_checkbox_column(self, index, header):
        pass

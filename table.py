from tkinter import *
from tkinter.ttk import *
import numpy as np


class TkinterTable:

    def __init__(self, root, table_data):
        # Assumes table data contains the headers as the first row.
        # Must be text data. Non-text data must be added via functions.
        self.root = root
        self.header_font = "Header.TLabel"
        self.num_columns = len(table_data[0])
        self.num_rows = len(table_data)
        header_style = Style()
        header_style.configure("Header.TLabel", font=('Segoe UI', 14, "bold"))

    def _get_section_width(self):
        pass

    def populate(self):
        pass

    def toggle_text_col(self, header):
        pass

    def redraw(self):
        pass

    def _configure_rows(self):
        for i in range(self.num_rows):
            self.root.rowconfigure(i, weight=1)

    def _configure_resize_weights(self):
        pass

    def _shift_n_columns(self, index, dir=1):
        pass

    def add_checkbox_column(self, index, header):
        pass

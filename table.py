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
                                         table_data[1:, i], i, header_style="Header.TLabel")]

    def get_num_cols(self):
        return len(self.columns)

    def get_num_rows(self):
        if len(self.columns) > 0:
            return self.columns[0].get_num_rows()
        return 0

    def _set_resize_widths(self, width):
        width = self.root.winfo_width()
        col_text_widths = []
        for c in self.columns:
            col_text_widths += [c.get_max_column_text_width()]

        total_text = sum(col_text_widths)
        screen_text_consumption = [width * col_text_widths[i] / total_text for i in range(len(col_text_widths))]
        for ind, c in enumerate(self.columns):
            # Assign resize weight as direct percentage of the total width it consumes
            self.root.columnconfigure(ind, weight=int(screen_text_consumption[ind]))
            c.set_wraptext_width(int(screen_text_consumption[ind]))

    def populate(self):
        for ind, c in enumerate(self.columns):
            c.add_to_parent_grid(ind)
        self.redraw(self.root.winfo_width())

    def redraw(self, width):
        self._set_resize_widths(width)
        self._configure_rows()

    def _configure_rows(self):
        for i in range(self.num_rows):
            self.root.rowconfigure(i, weight=1)

    def add_checkbox_column(self, index, header):
        for c in self.columns:
            if c.column_index >= index:
                c.shift_column(c.column_index + 1)
        self.columns = self.columns[:index] + [CheckboxColumn(self.root, header,
                                                              [False]*self.get_num_rows(), index,
                                                              header_style="Header.TLabel")] + self.columns[index:]
        self.columns[0].add_to_parent_grid(index)
        self.redraw(self.root.winfo_width())

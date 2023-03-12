from tkinter import *
from tkinter.ttk import *
import numpy as np

class TkinterTable:
    def __init__(self, root, table_data=None):
        # Assumes table data contains the headers as the first row.
        self.root = root
        self.table_data = table_data
        self.header_font = "Header.TLabel"
        self.num_sections = 0
        header_style = Style()
        header_style.configure("Header.TLabel", font=('Segoe UI', 14, "bold"))

    def _get_section_width(self):
        return self.root.winfo_width() / self.num_sections

    # Unused, but maybe useful later?
    def _delete_all_children(self):
        for item in self.root.winfo_children():
            item.destroy()

    def populate(self, table_data):
        self.table_data = table_data
        if not len(self.table_data) or not len(self.table_data[0]):
            return
        self.num_sections = len(self.table_data[0])

        for ind, header in enumerate(self.table_data[0]):
            e = Label(self.root, text=header, borderwidth=1, relief=GROOVE,
                      wraplength=self._get_section_width(), style=self.header_font)
            e.grid(column=ind, row=0, sticky=N+E+W)

        for row_ind, row in enumerate(self.table_data[1:]):
            for col_ind, val in enumerate(row):
                e = Label(self.root, text=str(val), borderwidth=1, relief=GROOVE,
                          wraplength=self._get_section_width())
                e.grid(column=col_ind, row=row_ind+1, sticky=N+S+E+W)  # Skipped headers, go up 1 row

        for ind in range(len(self.table_data)):
            self.root.rowconfigure(ind, weight=1)

        column_max_lengths = []
        for i in range(self.num_sections):
            col = self.table_data[:, i]
            mapped_col = map(lambda x: len(str(x)), col)
            column_max_lengths += [max(mapped_col)]
        column_max_length_total = sum(column_max_lengths)

        for i in range(self.num_sections):
            self.root.columnconfigure(i, weight=int((100*column_max_lengths[i]/column_max_length_total)))

    def repopulate(self):
        # Nuclear option
        # self._delete_all_children()
        # self.populate(self.table_data)
        for c in self.root.winfo_children():
            c.configure(wraplength=self._get_section_width())

    def add_column(self, index):
        self.num_sections += 1

    def remove_column(self, index):
        pass

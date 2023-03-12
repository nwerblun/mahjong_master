from tkinter import *
from tkinter.ttk import *
import numpy as np
from functools import partial


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

        self._configure_resize_weights()

    def redraw(self):
        # Nuclear option
        # self._delete_all_children()
        # self.populate(self.table_data)
        for c in self.root.winfo_children():
            if isinstance(c, Label):
                c.configure(wraplength=self._get_section_width())

    def _configure_resize_weights(self):
        column_max_lengths = []
        for i in range(self.num_sections):
            col = self.table_data[:, i]
            mapped_col = map(lambda x: len(str(x)), col)
            column_max_lengths += [max(mapped_col)]
        column_max_length_total = sum(column_max_lengths)

        for i in range(self.num_sections):
            self.root.columnconfigure(i, weight=int((100 * column_max_lengths[i] / column_max_length_total)))

    def _shift_n_columns_right(self, index):
        # Shift all columns starting from index to the right. index = 0 shifts all columns.
        for e in self.root.winfo_children():
            if e.grid_info()["column"] >= index:
                e.grid(column=e.grid_info()["column"]+1)

    def _checkbox_toggle(self, box_frame):
        info = box_frame.grid_info()
        row = info["row"]
        col = info["column"]
        val_to_insert = not (self.table_data[row][col] == "True")
        self.table_data[row][col] = str(val_to_insert)

    def add_checkbox_column(self, index, header):
        self.num_sections += 1
        self._shift_n_columns_right(index)
        # Header + false for every non-header row. Note these will be strings.
        data = np.array([header]+[False]*(len(self.table_data)-1))
        self.table_data = np.insert(self.table_data, index, data, axis=1)
        e = Label(self.root, text=header, borderwidth=1, relief=GROOVE,
                  wraplength=self._get_section_width(), style=self.header_font)
        e.grid(column=index, row=0, sticky=N+E+W)
        # Skip header row. Add 1 to row since we start at row 1
        for i in range(len(self.table_data)-1):
            # Add a frame so it can have a border
            f = Frame(self.root, borderwidth=1, relief=GROOVE)
            f.grid(column=index, row=i+1, sticky=N+S+E+W)
            # Cheat to make the lambda store the frm value when instantiated instead of the one at the end of the loop
            e = Checkbutton(f, command=(lambda frm=f: self._checkbox_toggle(frm)))
            e.pack(expand=YES)
            # Invoke twice to set it to off. Probably not the best way.
            e.invoke()
            e.invoke()
        self._configure_resize_weights()
        self.redraw()

    def remove_column(self, index):
        pass

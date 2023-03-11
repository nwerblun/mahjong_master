from tkinter import *


class TkinterTable:
    def __init__(self, root, table_data=None):
        self.root = root
        self.table_data = table_data

    def populate(self, table_data):
        self.table_data = table_data
        rows = len(self.table_data.keys()) + 1  # Include row for titles
        cols = len(list(self.table_data.values())[0])
        e = Label(self.root, text="Hand")
        e.grid(column=0, row=0)
        e = Label(self.root, text="Point Value")
        e.grid(column=1, row=0)
        e = Label(self.root, text="Categories")
        e.grid(column=2, row=0)
        e = Label(self.root, text="Notes")
        e.grid(column=3, row=0)
        curr_row = 1
        for key in self.table_data.keys():
            e = Label(self.root, text=key)
            e.grid(column=0, row=curr_row)
            for i in range(cols):
                e = Label(self.root, text=self.table_data[key][i])
                e.grid(column=i+1, row=curr_row)
            curr_row += 1


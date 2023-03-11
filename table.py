from tkinter import *


class TkinterTable:
    def __init__(self, root, table_data=None):
        self.root = root
        self.table_data = table_data

    def _get_section_width(self):
        return self.root.winfo_width() / 4

    # Deprecated
    def _delete_all_children(self):
        for item in self.root.winfo_children():
            item.destroy()

    def populate(self, table_data):
        self.table_data = table_data
        rows = len(self.table_data.keys()) + 1  # Include row for titles
        cols = len(list(self.table_data.values())[0])
        e = Label(self.root, text="Hand", justify=CENTER, wraplength=self._get_section_width())
        e.grid(column=0, row=0)
        e = Label(self.root, text="Point Value", justify=CENTER, wraplength=self._get_section_width())
        e.grid(column=1, row=0)
        e = Label(self.root, text="Categories", justify=CENTER, wraplength=self._get_section_width())
        e.grid(column=2, row=0)
        e = Label(self.root, text="Notes", justify=CENTER, wraplength=self._get_section_width())
        e.grid(column=3, row=0)
        curr_row = 1
        for key in self.table_data.keys():
            e = Label(self.root, text=key, justify=CENTER, wraplength=self._get_section_width())
            e.grid(column=0, row=curr_row)
            for i in range(cols):
                e = Label(self.root, text=self.table_data[key][i], justify=CENTER, wraplength=self._get_section_width())
                e.grid(column=i+1, row=curr_row)
            curr_row += 1

    def repopulate(self):
        # Nuclear option
        # self._delete_all_children()
        # self.populate(self.table_data)
        for c in self.root.winfo_children():
            c.configure(wraplength=self._get_section_width())

from tkinter import *
from tkinter.ttk import *
import numpy as np
from column import *


class TkinterTable:
    def __init__(self, root, table_data, image_prefix="[IMAGE]"):
        # Assumes table data contains the headers as the first row.
        # Must be text data. Non-text data must be added via functions.
        self.root = root
        self.header_font = "Header.TLabel"
        self.num_rows = len(table_data)
        self.columns = []
        self.image_prefix = image_prefix
        self.hidden_columns = {}
        self.sort_options = []
        self.sort_header = None
        self.exact_match_sort = False
        self.nothing_to_show_lbl = Label(self.root, text="Nothing found!")
        header_style = Style()
        header_style.configure("Header.TLabel", font=('Segoe UI', 14, "bold"))
        for i in range(len(table_data[0])):
            is_image = table_data[0, i][:len(self.image_prefix)] == self.image_prefix
            if is_image:
                self.columns += [LabelColumn(self.root, table_data[0, i][len(self.image_prefix):],
                                             table_data[1:, i], i, header_style="Header.TLabel", img=True)]
            else:
                self.columns += [LabelColumn(self.root, table_data[0, i],
                                             table_data[1:, i], i, header_style="Header.TLabel")]

    def get_col_data(self, header, search_hidden=False):
        total_cols = self.columns + list(self.hidden_columns.values()) if search_hidden else self.columns
        for c in total_cols:
            if c.header == header:
                return c.data
        return None

    def get_num_cols(self):
        return len(self.columns)

    def get_num_rows(self):
        if len(self.columns) > 0:
            return self.columns[0].get_num_rows()
        return 0

    def _set_resize_widths(self, width):
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

    def _add_column(self, col, index):
        self.columns = self.columns[:index] + [col] + self.columns[index:]

    def add_checkbox_column(self, index, header):
        index = len(self.columns) if index == -1 else index
        self._shift_columns_right(index)
        col = CheckboxColumn(self.root, header, [False]*self.get_num_rows(), index, header_style="Header.TLabel")
        self._add_column(col, index)
        self.columns[index].add_to_parent_grid(index)
        self.redraw(self.root.winfo_width())

    def add_label_column(self, index, header):
        index = len(self.columns) if index == -1 else index
        self._shift_columns_right(index)
        col = LabelColumn(self.root, header, [""] * self.get_num_rows(), index, header_style="Header.TLabel")
        self._add_column(col, index)
        self.columns[index].add_to_parent_grid(index)
        self.redraw(self.root.winfo_width())

    def toggle_column(self, header):
        all_cols = self.columns + list(self.hidden_columns.values())
        for c in all_cols:
            if c.header == header:
                index = c.column_index
                if c.hidden:
                    self._shift_columns_right(index)
                    self._add_column(self.hidden_columns[header], index)
                    del self.hidden_columns[header]
                    c.unhide()
                else:
                    c.hide()
                    self.hidden_columns[header] = c
                    self.columns.pop(index)
                    self._shift_columns_left(index)
        self.redraw(self.root.winfo_width())

    def toggle_sort_option(self, option, header):
        self.sort_header = header
        if option in self.sort_options:
            self.sort_options.pop(self.sort_options.index(option))
        else:
            self.sort_options += [option]
        self._populate_subset()

    def toggle_exact_match_sort(self):
        self.exact_match_sort = not self.exact_match_sort
        self._populate_subset()

    def _populate_subset(self):
        indices = []
        for c in (self.columns + list(self.hidden_columns.values())):
            if c.header == self.sort_header:
                for row, d in enumerate(c.data):
                    lowered = "".join(list(map(lambda x: x.lower(), d)))
                    contained_list = [tag in lowered for tag in self.sort_options]
                    if (not self.exact_match_sort) and any(contained_list):
                        indices += [row]
                    elif self.exact_match_sort and all(tag in lowered for tag in self.sort_options):
                        indices += [row]

        if len(indices) == 0 and self.exact_match_sort:
            for c in self.columns:
                c.grid_remove()
            self.nothing_to_show_lbl.grid(row=0, column=0, columnspan=len(self.columns))
            return
        else:
            self.nothing_to_show_lbl.grid_remove()
            for c in self.columns:
                c.grid_unremove()

        for c in self.columns:
            c.add_subset_to_parent_grid(indices)
        for v in self.hidden_columns.values():
            v.add_subset_to_parent_grid(indices)

    def _shift_columns_right(self, index):
        for c in self.columns[index:]:
            c.shift_column(c.column_index + 1)
        for c in self.hidden_columns.values():
            if c.column_index > index:
                c.shift_column(c.column_index + 1)

    def _shift_columns_left(self, index):
        for c in self.columns[index:]:
            c.shift_column(c.column_index - 1)
        for c in self.hidden_columns.values():
            if c.column_index > index:
                c.shift_column(c.column_index - 1)

import tkinter
from tkinter import *
from tkinter.ttk import *  # Automatically replace some widgets with better versions
from table import TkinterTable
from hands import MahjongHands
from utilities import *


class HandCalculator(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.image_prefix = "[IMAGE]"
        self.table_canvas = None
        self.canvas_container = None
        self.options_frame = None
        self.placeholder_frame2 = None
        self.sort_options_frame = None
        self.hands_table = None
        self.canvas_window = None
        self.reset_button_frame = None
        self._after_id = None
        self.show_hide_frame = None
        self.sort_buttons = []
        self.sort_button_cvs = []
        self.show_cat_button = None
        self.show_cat_button_cv = None
        self.show_img_button = None
        self.show_img_button_cv = None

    def _on_mousewheel(self, event):
        self.table_canvas.yview_scroll(-1 * (event.delta // 120), "units")

    def _on_canvas_table_config(self, event):
        if not (self._after_id is None):
            self.after_cancel(self._after_id)
        self._after_id = self.after(600, lambda w=event.width: self.hands_table.redraw(w))
        self.table_canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_canvas_frame_config(self, event):
        self.table_canvas.configure(scrollregion=self.table_canvas.bbox("all"))

    def _reset_sort_options(self):
        for i in range(len(self.sort_buttons)):
            if self.sort_button_cvs[i].get():
                self.sort_buttons[i].invoke()

    def _reset_all_options(self):
        recursive_destroy(self.canvas_container)
        recursive_destroy(self.options_frame)
        recursive_destroy(self.placeholder_frame2)
        self.canvas_container = None
        self.table_canvas = None
        self.options_frame = None
        self.placeholder_frame2 = None
        self.create_hand_table()

    def create_hand_table(self):
        # Top level frames. Placeholder
        self.placeholder_frame2 = Frame(self.master, borderwidth=3, relief=GROOVE, height=50, width=400)
        self.placeholder_frame2.pack(fill=X, side=TOP, pady=2, padx=2)
        # Center frame
        self.canvas_container = Frame(self.master, borderwidth=3, relief=GROOVE, height=300, width=400)
        self.canvas_container.pack(fill=BOTH, expand=YES, side=TOP, pady=2, padx=2)
        # Options frame
        self.options_frame = LabelFrame(self.master, text="Table Options", labelanchor="n",
                                        borderwidth=8, relief=GROOVE, height=150, width=400)
        self.options_frame.pack(fill=X, side=BOTTOM, pady=2, padx=2)

        # Create a canvas inside the center frame
        self.table_canvas = Canvas(self.canvas_container)
        # Add a scrollbar on the right and bottom
        vbar = Scrollbar(self.canvas_container, orient=VERTICAL)
        hbar = Scrollbar(self.canvas_container, orient=HORIZONTAL)
        vbar.pack(side=RIGHT, fill=Y)
        hbar.pack(side=BOTTOM, fill=X)
        # Set the bars' commands to modify the table's yview
        vbar.config(command=self.table_canvas.yview)
        hbar.config(command=self.table_canvas.xview)
        # Set the scroll command to modify the position of the vertical bar
        self.table_canvas.config(yscrollcommand=vbar.set)
        self.table_canvas.config(xscrollcommand=hbar.set)
        self.table_canvas.pack(expand=YES, fill=BOTH)

        # Don't need to pack a frame inside a canvas if it's going to be windowed.
        table_frame = Frame(self.table_canvas)
        # Create and populate hand table
        self.hands_table = TkinterTable(table_frame, MahjongHands.hands_info, self.image_prefix)
        self.hands_table.populate()

        # Set window to the center of the canvas and bind resize events
        x0 = self.table_canvas.winfo_width() / 2
        y0 = self.table_canvas.winfo_height() / 2
        self.canvas_window = self.table_canvas.create_window((x0, y0), window=table_frame)
        # Make sure that when we resize, the scrollable area is updated too and the frame is resized
        self.table_canvas.bind("<Configure>", self._on_canvas_table_config)
        table_frame.bind("<Configure>", self._on_canvas_frame_config)

        # Add the checkbox column and image column for hand condition
        image_col_header = MahjongHands.hands_info[0][4][len(self.image_prefix):]
        met_col_header = "Met?"
        self.hands_table.add_checkbox_column(0, met_col_header)

        # Add a frame to hold the reset button
        self.reset_button_frame = Frame(self.options_frame, borderwidth=1, relief=GROOVE)
        self.reset_button_frame.pack(side=RIGHT, anchor="e", fill=BOTH)

        # Add a frame to hold sort options
        self.sort_options_frame = LabelFrame(self.options_frame, text="Sort Options", labelanchor="n",
                                             borderwidth=1, relief=GROOVE)
        self.sort_options_frame.pack(side=RIGHT, expand=YES, fill=BOTH, anchor="w")

        # Add checkbuttons to sort frame based on categories
        # Combine all categories into one mega string comma separated, no spaces, ultra python syntax, incomprehensible
        category_header = MahjongHands.hands_info[0][2]
        all_categories_list = [c for s in MahjongHands.hands_info[1:, 2]
                               for c in map(lambda x: x.strip(), s.split(","))]
        # Get only unique entries
        all_categories_list = sorted(list(set(all_categories_list)))
        self.sort_buttons = []
        self.sort_button_cvs = []
        for ind, cat in enumerate(all_categories_list):
            self.sort_button_cvs += [IntVar(self.sort_options_frame)]
            self.sort_button_cvs[ind].set(0)
            self.sort_buttons += [Checkbutton(self.sort_options_frame, text=cat, variable=self.sort_button_cvs[ind])]
            self.sort_buttons[ind].pack(side=LEFT, expand=YES, fill=BOTH)
            self.sort_buttons[ind].configure(
                command=lambda x=cat: self.hands_table.toggle_sort_option(x, category_header))

        # Add a frame for show/hide options
        self.show_hide_frame = LabelFrame(self.options_frame, text="Show/Hide", labelanchor="n",
                                          borderwidth=1, relief=GROOVE)
        self.show_hide_frame.pack(side=LEFT, anchor="n", fill=BOTH)

        # Place checkboxes in the options frame
        self.show_cat_button_cv = IntVar(self.show_hide_frame)
        self.show_cat_button_cv.set(1)
        self.show_cat_button = Checkbutton(
            self.show_hide_frame, text="Show categories", variable=self.show_cat_button_cv)
        self.show_cat_button.pack(side=TOP, anchor="n")
        self.show_cat_button.configure(command=lambda h=category_header: self.hands_table.toggle_column(h))
        self.show_cat_button.invoke()

        self.show_img_button_cv = IntVar(self.show_hide_frame)
        self.show_img_button_cv.set(1)
        self.show_img_button = Checkbutton(
            self.show_hide_frame, text=image_col_header, variable=self.show_img_button_cv)
        self.show_img_button.pack(side=BOTTOM, anchor="s")
        self.show_img_button.configure(command=lambda h=image_col_header: self.hands_table.toggle_column(h))

        rst_all_btn = Button(self.reset_button_frame, text="Reset All Selections", command=self._reset_all_options)
        rst_all_btn.pack(side=TOP, expand=YES, fill=BOTH)
        rst_sort_btn = Button(self.reset_button_frame, text="Reset Sort Selections", command=self._reset_sort_options)
        rst_sort_btn.pack(side=BOTTOM, expand=YES, fill=BOTH)

        # Scroll canvas to the top after everything is done
        self.table_canvas.after(1000, self.table_canvas.yview_scroll, -1000, "units")
        bind_all_children(self.canvas_container, "<MouseWheel>", self._on_mousewheel)



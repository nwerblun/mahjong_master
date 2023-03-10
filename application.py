import tkinter
from tkinter import *
from tkinter.ttk import *  # Automatically replace some widgets with better versions
from table import TkinterTable
from hands import MahjongHands


class Core(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.table_canvas = None
        self.canvas_container = None
        self.options_frame = None
        self.placeholder_frame2 = None
        self.hands_table = None
        self.canvas_window = None
        self._after_id = None
        self.create_hand_table()
        self.bind_all_children(self.canvas_container, "<MouseWheel>", self._on_mousewheel)

    @staticmethod
    def bind_all_children(parent, event_name, func):
        for c in parent.winfo_children():
            c.bind(event_name, func)
            Core.bind_all_children(c, event_name, func)

    def _on_mousewheel(self, event):
        self.table_canvas.yview_scroll(-1 * (event.delta // 120), "units")

    def _on_canvas_table_config(self, event):
        if not (self._after_id is None):
            self.after_cancel(self._after_id)
        self._after_id = self.after(600, lambda: self.hands_table.redraw())
        self.table_canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_canvas_frame_config(self, event):
        self.table_canvas.configure(scrollregion=self.table_canvas.bbox("all"))

    def create_hand_table(self):
        # Top level frames. Placeholder
        self.placeholder_frame2 = Frame(self.master, borderwidth=3, relief=GROOVE, height=50, width=400)
        self.placeholder_frame2.pack(fill=X, side=TOP, pady=2, padx=2)
        # Center frame
        self.canvas_container = Frame(self.master, borderwidth=3, relief=GROOVE, height=300, width=400)
        self.canvas_container.pack(fill=BOTH, expand=YES, side=TOP, pady=2, padx=2)
        # Options frame
        self.options_frame = LabelFrame(self.master, text="Table Options", labelanchor="n",
                                        borderwidth=3, relief=GROOVE, height=150, width=400)
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
        self.hands_table = TkinterTable(table_frame, MahjongHands.hands_info)
        self.hands_table.populate()
        # TODO: Refactor the table class
        # TODO: Check the rest of this
        # Place checkboxes in the options frame
        # category_header = MahjongHands.hands_info[0][2]
        # e = Checkbutton(self.options_frame, text="Show categories",
        #                 command=lambda h=category_header: self.hands_table.toggle_text_col(h))
        # e.pack(side=TOP, anchor="nw")
        # e.invoke()
        # e.invoke()

        # Set window to the center of the canvas
        x0 = self.table_canvas.winfo_width() / 2
        y0 = self.table_canvas.winfo_height() / 2
        self.canvas_window = self.table_canvas.create_window((x0, y0), window=table_frame)
        # Make sure that when we resize, the scrollable area is updated too and the frame is resized
        self.table_canvas.bind("<Configure>", self._on_canvas_table_config)
        table_frame.bind("<Configure>", self._on_canvas_frame_config)
        # Add the checkbox column for hand condition
        # self.hands_table.add_checkbox_column(0, "Met?")
        self.table_canvas.after(1000, self.table_canvas.yview_scroll, -1000, "units")



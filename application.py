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
        self.placeholder_frame = None
        self.placeholder_frame2 = None
        self.hands_table = None
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
        # Another placeholder
        self.placeholder_frame = Frame(self.master, borderwidth=3, relief=GROOVE, height=50, width=400)
        self.placeholder_frame.pack(fill=X, side=BOTTOM, pady=2, padx=2)

        # Create a canvas inside the center frame
        self.table_canvas = Canvas(self.canvas_container, background="green", borderwidth=5, relief=SUNKEN)
        # Add a scrollbar on the right
        vbar = Scrollbar(self.canvas_container, orient=VERTICAL)
        vbar.pack(side=RIGHT, fill=Y)
        vbar.config(command=self.table_canvas.yview)
        self.table_canvas.config(yscrollcommand=vbar.set)
        self.table_canvas.pack(expand=YES, fill=BOTH)

        # Test frame full of labels
        s = Style()
        s.configure('Test.TFrame', background='maroon')
        # Don't need to pack a frame in a canvas
        self.test_frame = Frame(self.table_canvas, style="Test.TFrame")
        # Add 30 test labels
        for i in range(30):
            x = Label(self.test_frame, borderwidth=5, relief=GROOVE, text=str(i)*5)
            x.grid(row=i, column=i//5)

        x0 = self.table_canvas.winfo_width() / 2
        y0 = self.table_canvas.winfo_height() / 2
        self.canvas_window = self.table_canvas.create_window((x0, y0), window=self.test_frame)
        # Make sure that when we resize, the scrollable area is updated too and the frame is resized
        self.table_canvas.bind("<Configure>", self._on_canvas_table_config)
        self.test_frame.bind("<Configure>", self._on_canvas_frame_config)
        # table_frame = Frame(self.table_canvas, height=300, width=400)
        # table_frame.pack(fill=BOTH, expand=YES)
        #
        # self.hands_table = TkinterTable(table_frame)
        # self.hands_table.populate(MahjongHands.hands_info)




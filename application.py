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
        self.bind_all_children(self.canvas_container, "<MouseWheel>", self.on_mousewheel)

    @staticmethod
    def bind_all_children(parent, event_name, func):
        for c in parent.winfo_children():
            c.bind(event_name, func)
            Core.bind_all_children(c, event_name, func)

    def on_mousewheel(self, event):
        self.table_canvas.yview_scroll(-1 * (event.delta // 120), "units")

    def create_hand_table(self):
        # Top level frames
        self.placeholder_frame2 = Frame(self.master, borderwidth=3, relief=GROOVE, height=50, width=400)
        self.placeholder_frame2.pack(fill=X, side=TOP, pady=2, padx=2)

        self.canvas_container = Frame(self.master, borderwidth=3, relief=GROOVE, height=300, width=400)
        self.canvas_container.pack(fill=BOTH, expand=YES, side=TOP, pady=2, padx=2)

        self.placeholder_frame = Frame(self.master, borderwidth=3, relief=GROOVE, height=50, width=400)
        self.placeholder_frame.pack(fill=X, side=BOTTOM, pady=2, padx=2)
        # Create a canvas inside the center frame
        self.table_canvas = Canvas(self.canvas_container, borderwidth=5, relief=SUNKEN)
        vbar = Scrollbar(self.canvas_container, orient=VERTICAL)
        vbar.pack(side=RIGHT, fill=Y)
        vbar.config(command=self.table_canvas.yview)
        self.table_canvas.config(yscrollcommand=vbar.set)
        self.table_canvas.pack(fill=BOTH, expand=YES)

        test_frame = Frame(self.table_canvas, borderwidth=5, relief=RIDGE)
        test_frame.pack(expand=YES, fill=BOTH)
        for i in range(20):
            x = Label(test_frame, borderwidth=5, relief=GROOVE, text=str(i)*5)
            x.grid(row=i, column=0, sticky=N+E+S+W)
        self.table_canvas.create_window(0, 0, window=test_frame)
        # table_frame = Frame(self.table_canvas, height=300, width=400)
        # table_frame.pack(fill=BOTH, expand=YES)
        #
        # self.hands_table = TkinterTable(table_frame)
        # self.hands_table.populate(MahjongHands.hands_info)

        # self.table_canvas.create_window(0, 0, window=table_frame)
        self.table_canvas.configure(scrollregion=self.table_canvas.bbox("all"))




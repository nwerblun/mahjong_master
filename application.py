from tkinter import *
from tkinter.ttk import *  # Automatically replace some widgets with better versions
from hand_calculator import HandCalculator
from hand_assister import HandAssister
from utilities import recursive_destroy


class Application(Frame):
    def __init__(self, root, *args, **kwargs):
        Frame.__init__(self, root, *args, **kwargs)
        self.hand_calc = None
        self.hand_assister = None
        self.hand_helper_frame = None
        self.hand_analyzer_real_time_frame = None
        self.pack(expand=YES, fill=BOTH)

        self.tab_text_style = Style()
        self.tab_text_style.configure("TNotebook.Tab", font=('Segoe UI', 12))
        self.notebook = Notebook(self)
        self.notebook.pack(expand=YES, fill=BOTH)

        self.hand_calc_frame = Frame(self.notebook)
        self.hand_calc_frame.pack(fill=BOTH, expand=YES)
        self.create_hand_calc(self.hand_calc_frame)
        self.notebook.add(self.hand_calc_frame, text="Hand Info and Calculator")

        self.hand_helper_frame = Frame(self.notebook)
        self.hand_helper_frame.pack(fill=BOTH, expand=YES)
        self.create_hand_helper(self.hand_helper_frame)
        self.notebook.add(self.hand_helper_frame, text="Hand Helper")

        self.hand_analyzer_real_time_frame = Frame(self.notebook)
        self.hand_analyzer_real_time_frame.pack(fill=BOTH, expand=YES)
        self.temp_lbl = Label(self.hand_analyzer_real_time_frame, text="UNDER CONSTRUCTION")
        self.temp_lbl.pack(fill=BOTH, expand=YES)
        self.notebook.add(self.hand_analyzer_real_time_frame, text="Real-time Hand Analyzer")
        # self.debug()

    def create_hand_calc(self, root):
        self.hand_calc = HandCalculator(root)
        self.hand_calc.create_hand_table()

    def create_hand_helper(self, root):
        self.hand_assister = HandAssister(root)
        self.hand_assister.create_hand_entry()
        self.hand_assister.create_hand_visualizer()

    def destroy_hand_calc(self):
        if self.hand_calc is not None:
            recursive_destroy(self.hand_calc_frame)
            self.hand_calc_frame = None
            self.hand_calc = None

    # def debug(self):
    #     self.hand_assister.concealed_other_entries[0].insert(0, "")
    #     self.hand_assister.concealed_other_entries[1].insert(0, "c9")
    #     self.hand_assister.concealed_other_entries[2].insert(0, "d3")
    #     self.hand_assister.concealed_other_entries[3].insert(0, "d7")
    #     self.hand_assister.concealed_other_entries[4].insert(0, "d8")
    #     self.hand_assister.concealed_other_entries[5].insert(0, "drr")
    #     self.hand_assister.concealed_other_entries[6].insert(0, "drg")
    #     self.hand_assister.concealed_other_entries[7].insert(0, "")
    #     self.hand_assister.concealed_other_entries[8].insert(0, "")
    #     self.hand_assister.concealed_other_entries[9].insert(0, "")
    #
    #     self.hand_assister.revealed_other_entries[0].insert(0, "d4")
    #     self.hand_assister.revealed_other_entries[1].insert(0, "d5")
    #     self.hand_assister.revealed_other_entries[2].insert(0, "d6")
    #
    #     self.hand_assister.revealed_kong_entries[0].insert(0, "wn")
    #
    #     self.hand_assister.final_tile_entry.insert(0, "c")
    #     self.hand_assister.final_tile_drawn_or_discard_checkbutton.invoke()


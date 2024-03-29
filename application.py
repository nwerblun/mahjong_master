from tkinter import *
from tkinter.ttk import *  # Automatically replace some widgets with better versions
from hand_calculator import HandCalculator
from hand_assister import HandAssister
from real_time_hand_analyzer import HandAnalyzer
from utilities import recursive_destroy


class Application(Frame):
    def __init__(self, root, *args, **kwargs):
        Frame.__init__(self, root, *args, **kwargs)
        self.hand_calc = None
        self.hand_assister = None
        self.analyzer = None
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
        self.create_real_time_analyzer(self.hand_analyzer_real_time_frame)
        self.notebook.add(self.hand_analyzer_real_time_frame, text="Real-time Hand Analyzer")

    def create_hand_calc(self, root):
        self.hand_calc = HandCalculator(root)
        self.hand_calc.create_hand_table()

    def create_real_time_analyzer(self, root):
        self.analyzer = HandAnalyzer(root)
        self.analyzer.create_application_selector()
        self.analyzer.create_application_preview()
        self.analyzer.create_auto_hand_visualizer()
        self.analyzer.create_solver_area()

    def create_hand_helper(self, root):
        self.hand_assister = HandAssister(root)
        self.hand_assister.create_hand_entry()
        self.hand_assister.create_hand_visualizer()

    def destroy_hand_calc(self):
        if self.hand_calc is not None:
            recursive_destroy(self.hand_calc_frame)
            self.hand_calc_frame = None
            self.hand_calc = None



import tkinter
from tkinter import *
from tkinter.ttk import *  # Automatically replace some widgets with better versions
from hands import MahjongHands
from utilities import *

# Similar to hand_calculator.
# Generates the frames and GUI to hold the hand entry, visualizer and nearest hands calculator


class HandAssister(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.hand_entry_frame = Frame(self.master, borderwidth=3, relief=GROOVE)
        self.hand_entry_frame.pack(side=TOP, fill=BOTH)
        self.instructions_text = None
        self.concealed_entry_labelframe = None
        self.revealed_entry_labelframe = None
        self.concealed_kong_entries = []
        self.concealed_kong_entries_cvs = []
        self.concealed_other_entries = []
        self.concealed_other_entries_cvs = []
        self.revealed_kong_entries = []
        self.revealed_kong_entries_cvs = []
        self.revealed_other_entries = []
        self.revealed_other_entries_cvs = []
        self.concealed_kong_entry_label = None
        self.concealed_others_entry_label = None
        self.revealed_others_entry_label = None
        self.revealed_kong_entry_label = None
        self.entry_validation = None

    @staticmethod
    def _check_valid_hand_entry(text):
        invalid_conds = [
            len(text) == 1 and not (text in [name[:1] for name in MahjongHands.tile_names]),
            len(text) == 2 and not (text in [name[:2] for name in MahjongHands.tile_names]),
            len(text) == 3 and not (text in MahjongHands.tile_names),
            len(text) > 3
        ]
        if any(invalid_conds):
            return False
        return True

    def create_hand_entry(self):
        self.entry_validation = self.register(HandAssister._check_valid_hand_entry)
        instructions = "Instructions:\n" \
                       "Enter your hand, one tile per box (excluding kongs. Just type the tile once for a kong)\n" \
                       "b1-9 = Bamboo. Ex: b1 b5\n" \
                       "c1-9 = Characters. Ex c4 c8\n" \
                       "d1-9 = Dots. Ex d5 d6\n" \
                       "drw, drr, drg = White/Red/Green Dragon\n" \
                       "wn, we, ws, ww = North/East/South/West Wind\n"
        self.instructions_text = Message(self.hand_entry_frame, text=instructions, aspect=400)
        self.instructions_text.grid(rowspan=2, column=0, sticky=W+N+S)

        self.concealed_entry_labelframe = LabelFrame(self.hand_entry_frame, text="Concealed Tiles")
        self.concealed_entry_labelframe.grid(row=0, column=1, sticky=N+E+S+W)
        self.revealed_entry_labelframe = Labelframe(self.hand_entry_frame, text="Revealed Tiles")
        self.revealed_entry_labelframe.grid(row=1, column=1, sticky=N+E+S+W)

        self.concealed_kong_entry_label = Label(self.concealed_entry_labelframe, text="Kongs:")
        self.concealed_kong_entry_label.grid(row=0, column=0, sticky=N+S)
        for i in range(4):
            self.concealed_kong_entries_cvs += [StringVar()]
            self.concealed_kong_entries += [Entry(self.concealed_entry_labelframe, validate="key",
                                                  width=4, textvariable=self.concealed_kong_entries_cvs[i],
                                                  validatecommand=(self.entry_validation, '%P'))]
            self.concealed_kong_entries[i].grid(row=0, column=i+1, padx=4, sticky=N+S)

        self.concealed_others_entry_label = Label(self.concealed_entry_labelframe, text="Others:")
        self.concealed_others_entry_label.grid(row=1, column=0, pady=2, sticky=N+S)
        for i in range(14):
            self.concealed_other_entries_cvs += [StringVar()]
            self.concealed_other_entries += [Entry(self.concealed_entry_labelframe, validate="key",
                                                   width=4, textvariable=self.concealed_other_entries_cvs[i],
                                                   validatecommand=(self.entry_validation, '%P'))]
            self.concealed_other_entries[i].grid(row=1, column=i+1, pady=2, padx=2, sticky=N+S)

        self.revealed_kong_entry_label = Label(self.revealed_entry_labelframe, text="Kongs:")
        self.revealed_kong_entry_label.grid(row=0, column=0, sticky=N+S)
        for i in range(4):
            self.revealed_kong_entries_cvs += [StringVar()]
            self.revealed_kong_entries += [Entry(self.revealed_entry_labelframe, validate="key", width=4,
                                                 textvariable=self.revealed_kong_entries_cvs[i],
                                                 validatecommand=(self.entry_validation, '%P'))]
            self.revealed_kong_entries[i].grid(row=0, column=i+1, padx=4, sticky=N+S)

        self.revealed_others_entry_label = Label(self.revealed_entry_labelframe, text="Others:")
        self.revealed_others_entry_label.grid(row=1, column=0, pady=2, sticky=N+S)
        for i in range(14):
            self.revealed_other_entries_cvs += [StringVar()]
            self.revealed_other_entries += [Entry(self.revealed_entry_labelframe, validate="key", width=4,
                                                  textvariable=self.revealed_other_entries_cvs[i],
                                                  validatecommand=(self.entry_validation, '%P'))]
            self.revealed_other_entries[i].grid(row=1, column=i+1, pady=2, padx=2, sticky=N+S)



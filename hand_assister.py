from tkinter import *
from tkinter.ttk import *  # Automatically replace some widgets with better versions
from hands import MahjongHands
from utilities import *
from score_calculator import Calculator


# TODO: Add round + seat wind entry
# TODO: Add a note saying revealed tiles should be in set order
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
        self.drawn_tile_labelframe = None
        self.drawn_tile_entry = None
        self.drawn_tile_entry_cv = None
        self.entry_validation = None
        self.calculator = Calculator()

    def _check_valid_hand_entry(self, text):
        invalid_conds = [
            len(text) == 1 and not (text in [name[:1] for name in MahjongHands.tile_names]),
            len(text) == 2 and not (text in [name[:2] for name in MahjongHands.tile_names]),
            len(text) == 3 and not (text in MahjongHands.tile_names),
            len(text) > 3
        ]
        if any(invalid_conds):
            return False
        self.after(1, self._hand_change)
        return True

    def _hand_change(self):
        concealed = []
        revealed = []
        drawn = self.drawn_tile_entry_cv.get()
        concealed_kongs = []
        revealed_kongs = []
        for e in self.concealed_other_entries_cvs:
            if len(e.get()) == 2 or len(e.get()) == 3:
                concealed += [e.get()]
        for e in self.concealed_kong_entries_cvs:
            if len(e.get()) == 2 or len(e.get()) == 3:
                concealed_kongs += [e.get()]
        for e in self.revealed_other_entries_cvs:
            if len(e.get()) == 2 or len(e.get()) == 3:
                revealed += [e.get()]
        for e in self.revealed_kong_entries_cvs:
            if len(e.get()) == 2 or len(e.get()) == 3:
                revealed_kongs += [e.get()]
        self.calculator.set_hand(concealed, revealed, drawn, concealed_kongs, revealed_kongs)

    def create_hand_entry(self):
        self.entry_validation = self.register(self._check_valid_hand_entry)
        self.hand_entry_frame.rowconfigure("all", weight=1)
        instructions = "Instructions:\n" \
                       "Enter your hand, one tile per box (Excluding kongs. Just type the tile once for a kong)\n" \
                       "If you have 4x a tile concealed that is undeclared, enter it in 'others' instead of kong\n" \
                       "IMPORTANT: Enter revealed tiles in order or things wont work. EX: " \
                       "if you have 111222333 bamboo then either input it as "\
                       "b1b2b3 b1b2b3 b1b2b3 or b1b1b1 b2b2b2 b3b3b3. Do not mix up the order or put it in weird." \
                       "===========================================" \
                       "b1-9 = Bamboo. Ex: b1 b5\n" \
                       "c1-9 = Characters. Ex c4 c8\n" \
                       "d1-9 = Dots. Ex d5 d6\n" \
                       "drw, drr, drg = White/Red/Green Dragon\n" \
                       "wn, we, ws, ww = North/East/South/West Wind\n"
        self.instructions_text = Message(self.hand_entry_frame, text=instructions, aspect=400)
        self.instructions_text.grid(row=0, rowspan=2, column=0, sticky=W+N+S)

        self.drawn_tile_labelframe = Labelframe(self.hand_entry_frame, text="Drawn Tile")
        self.drawn_tile_labelframe.grid(row=0, rowspan=2, column=1, padx=15, pady=2, sticky=N+S)
        self.drawn_tile_entry_cv = StringVar()
        self.drawn_tile_entry = Entry(self.drawn_tile_labelframe, validate="key", width=4,
                                      textvariable=self.drawn_tile_entry_cv,
                                      validatecommand=(self.entry_validation, '%P'))
        self.drawn_tile_entry.pack(expand=YES)
        self.concealed_entry_labelframe = LabelFrame(self.hand_entry_frame, text="Concealed Tiles")
        self.concealed_entry_labelframe.grid(row=0, column=2, pady=8, sticky=N+E+S+W)
        self.revealed_entry_labelframe = Labelframe(self.hand_entry_frame, text="Revealed Tiles")
        self.revealed_entry_labelframe.grid(row=1, column=2, pady=8, sticky=N+E+S+W)

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

    def create_hand_visualizer(self):
        pass

    # TODO: Remove this
    def _debug_func(self):
        self.drawn_tile_entry.insert(0, "d9")
        self.concealed_kong_entries[0].insert(0, "b7")
        self.concealed_kong_entries[1].insert(0, "")
        self.concealed_kong_entries[2].insert(0, "")
        self.concealed_kong_entries[3].insert(0, "")

        self.revealed_kong_entries[0].insert(0, "")
        self.revealed_kong_entries[1].insert(0, "")
        self.revealed_kong_entries[2].insert(0, "")
        self.revealed_kong_entries[3].insert(0, "")

        self.concealed_other_entries[0].insert(0, "b1")
        self.concealed_other_entries[1].insert(0, "b2")
        self.concealed_other_entries[2].insert(0, "b2")
        self.concealed_other_entries[3].insert(0, "b2")
        self.concealed_other_entries[4].insert(0, "b2")
        self.concealed_other_entries[5].insert(0, "b3")
        self.concealed_other_entries[6].insert(0, "b4")
        self.concealed_other_entries[7].insert(0, "")
        self.concealed_other_entries[8].insert(0, "")
        self.concealed_other_entries[9].insert(0, "")
        self.concealed_other_entries[10].insert(0, "")
        self.concealed_other_entries[11].insert(0, "")
        self.concealed_other_entries[12].insert(0, "")
        self.concealed_other_entries[13].insert(0, "")

        self.revealed_other_entries[0].insert(0, "c2")
        self.revealed_other_entries[1].insert(0, "c3")
        self.revealed_other_entries[2].insert(0, "c4")
        self.revealed_other_entries[3].insert(0, "")
        self.revealed_other_entries[4].insert(0, "")
        self.revealed_other_entries[5].insert(0, "")
        self.revealed_other_entries[6].insert(0, "")
        self.revealed_other_entries[7].insert(0, "")
        self.revealed_other_entries[8].insert(0, "")
        self.revealed_other_entries[9].insert(0, "")
        self.revealed_other_entries[10].insert(0, "")
        self.revealed_other_entries[11].insert(0, "")
        self.revealed_other_entries[12].insert(0, "")
        self.revealed_other_entries[13].insert(0, "")

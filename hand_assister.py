from tkinter import *
from tkinter.ttk import *  # Automatically replace some widgets with better versions
from hands import MahjongHands
from utilities import *
from score_calculator import Calculator
from game import VoidTile


class HandAssister(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.hand_entry_frame = Frame(self.master, borderwidth=3, relief=GROOVE)
        self.hand_entry_frame.pack(side=TOP, fill=BOTH)
        self.void_tile = VoidTile()
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
        self.round_seat_wind_frame = None
        self.round_wind_dropdown = None
        self.seat_wind_dropdown = None
        self.seat_wind_cv = None
        self.round_wind_cv = None
        self.clear_hand_button = None
        self.hand_visualizer_frame = None
        self.visualizer_revealed_frame = None
        self.visualizer_concealed_frame = None
        self.visualizer_dawn_tile_frame = None
        self.visualizer_revealed_set_label = None
        self.visualizer_concealed_set_label = None
        self.visualizer_drawn_tile_label = None
        self.visualizer_revealed_set_tile_pictures = []
        self.visualizer_concealed_set_tile_pictures = []
        self.visualizer_drawn_tile_picture = None
        self.hand_entry_warning_label = None
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
        self._update_visualizer()

    def _update_visualizer(self):
        for l in self.visualizer_revealed_set_tile_pictures:
            l.pack_forget()
        for l in self.visualizer_concealed_set_tile_pictures:
            l.pack_forget()
        self.visualizer_drawn_tile_picture.pack_forget()

        concealed = self.calculator.hand.concealed_tiles[:16]
        declared_kongs = self.calculator.hand.declared_concealed_kongs[:4]
        c, p, k = self.calculator.hand.get_revealed_sets()
        drawn = self.calculator.hand.drawn_tile
        for i, t in enumerate(concealed):
            self.visualizer_concealed_set_tile_pictures[i].configure(image=t.ph)
            self.visualizer_concealed_set_tile_pictures[i].pack(side=LEFT, anchor="w")

        lbl_ctr = 0
        for r_set in c+p+k:
            if lbl_ctr + len(r_set) <= 16:
                for i in range(len(r_set) - 1):
                    self.visualizer_revealed_set_tile_pictures[lbl_ctr].configure(image=r_set[i].ph)
                    self.visualizer_revealed_set_tile_pictures[lbl_ctr].pack(side=LEFT, anchor="w")
                    lbl_ctr += 1
                if len(r_set) > 1:
                    self.visualizer_revealed_set_tile_pictures[lbl_ctr].configure(image=r_set[-1].ph)
                    self.visualizer_revealed_set_tile_pictures[lbl_ctr].pack(side=LEFT, padx=(0, 18), anchor="w")
                    lbl_ctr += 1

        start = False
        for d_set in declared_kongs:
            if lbl_ctr + len(d_set) <= 16:
                for i in range(len(d_set) - 1):
                    if not start:
                        self.visualizer_revealed_set_tile_pictures[lbl_ctr].configure(image=self.void_tile.ph)
                        self.visualizer_revealed_set_tile_pictures[lbl_ctr].pack(side=LEFT, anchor="w")
                        start = True
                    else:
                        self.visualizer_revealed_set_tile_pictures[lbl_ctr].configure(image=d_set[i].ph)
                        self.visualizer_revealed_set_tile_pictures[lbl_ctr].pack(side=LEFT, anchor="w")
                    lbl_ctr += 1
                if len(d_set) > 1:
                    self.visualizer_revealed_set_tile_pictures[lbl_ctr].configure(image=self.void_tile.ph)
                    self.visualizer_revealed_set_tile_pictures[lbl_ctr].pack(side=LEFT, padx=(0, 18), anchor="w")
                    lbl_ctr += 1
                start = False

        if drawn:
            self.visualizer_drawn_tile_picture.configure(image=drawn.ph)
            self.visualizer_drawn_tile_picture.pack(expand=YES, fill=BOTH)

    def _clear_hand_entry(self):
        for p in self.visualizer_concealed_set_tile_pictures:
            p.pack_forget()
        for p in self.visualizer_revealed_set_tile_pictures:
            p.pack_forget()
        self.visualizer_drawn_tile_picture.pack_forget()

        for e in self.concealed_other_entries:
            e.delete(0, 3)
        for e in self.concealed_kong_entries:
            e.delete(0, 3)
        for e in self.revealed_other_entries:
            e.delete(0, 3)
        for e in self.revealed_kong_entries:
            e.delete(0, 3)
        self.drawn_tile_entry.delete(0, 3)

    # TODO: add 'final tile from wall/discard button'??? or a note saying it's ignored
    def create_hand_entry(self):
        self.entry_validation = self.register(self._check_valid_hand_entry)
        self.hand_entry_frame.rowconfigure("all", weight=1)
        self.hand_entry_frame.columnconfigure(0, weight=1)
        self.hand_entry_frame.columnconfigure(1, weight=1)
        self.hand_entry_frame.columnconfigure(2, weight=3)
        self.hand_entry_frame.columnconfigure(3, weight=5)
        instructions = "Instructions:\n" \
                       "Enter your hand, one tile per box\n" \
                       "(Excluding kongs. Just type the tile once for a kong)\n" \
                       "If you have 4x of a tile concealed that is undeclared,\n" \
                       "enter it in 'others' instead of kong\n" \
                       "==========================================\n" \
                       "If you steal a tile, just put your new set into the revealed section\n" \
                       "and leave 'drawn tile' blank.\n" \
                       "===========================================\n" \
                       "b1-9 = Bamboo. Ex: b1 b5\n" \
                       "c1-9 = Characters. Ex c4 c8\n" \
                       "d1-9 = Dots. Ex d5 d6\n" \
                       "drw, drr, drg = White/Red/Green Dragon\n" \
                       "wn, we, ws, ww = North/East/South/West Wind\n"
        self.instructions_text = Message(self.hand_entry_frame, text=instructions, aspect=400)
        self.instructions_text.grid(row=0, rowspan=3, column=0, sticky=W+N+S)

        self.drawn_tile_labelframe = Labelframe(self.hand_entry_frame, text="Drawn Tile")
        self.drawn_tile_labelframe.grid(row=0, rowspan=3, column=1, padx=15, pady=2, sticky=N+S)
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

        warning_text = "IMPORTANT: Enter revealed tile sets in order or things wont work." \
                       " Also note that your tiles may" \
                       " disappear while you are typing. This is normal.\n" \
                       "EXAMPLE: if you have b1 b1 b1 b2 b2 b2 b3 b3 b3 it does matter if you type it as\n" \
                       "b1b2b3 b1b2b3 b1b2b3 VS. b1b1b1 b2b2b2 b3b3b3\n" \
                       "Try not to mix up the order or put it in weird. :)"
        self.hand_entry_warning_label = Message(self.hand_entry_frame, text=warning_text, aspect=1200)
        self.hand_entry_warning_label.grid(row=2, column=2, sticky=W)

        self.clear_hand_button = Button(self.hand_entry_frame, text="Clear Hand", command=self._clear_hand_entry)
        self.clear_hand_button.grid(row=3, column=2, sticky=W+E)

        self.round_seat_wind_frame = Frame(self.hand_entry_frame)
        self.round_seat_wind_frame.grid(row=0, rowspan=3, column=3, padx=15, sticky=N+S)
        e = Label(self.round_seat_wind_frame, text="Round Wind:")
        e.pack(side=TOP, pady=4, anchor="w")
        self.round_wind_cv = StringVar()
        self.round_wind_dropdown = Combobox(self.round_seat_wind_frame, textvariable=self.round_wind_cv,
                                            exportselection=False, justify=LEFT,
                                            values=("North", "East", "South", "West"))
        self.round_wind_dropdown.pack(side=TOP, expand=YES, fill=X, pady=4, anchor="n")
        self.round_wind_dropdown.state(['!disabled', 'readonly'])
        self.round_wind_cv.set("East")
        e = Label(self.round_seat_wind_frame, text="Seat Wind:")
        e.pack(side=TOP, pady=4, anchor="w")
        self.seat_wind_cv = StringVar()
        self.seat_wind_dropdown = Combobox(self.round_seat_wind_frame, textvariable=self.seat_wind_cv,
                                           exportselection=False, justify=LEFT,
                                           values=("North", "East", "South", "West"))
        self.seat_wind_dropdown.pack(side=TOP, expand=YES, fill=X, pady=4, anchor="n")
        self.seat_wind_cv.set("East")
        self.seat_wind_dropdown.state(['!disabled', 'readonly'])

    def create_hand_visualizer(self):
        self.hand_visualizer_frame = LabelFrame(self.hand_entry_frame, text="Your Hand", labelanchor="n")
        self.hand_visualizer_frame.grid(row=0, rowspan=3, padx=15, pady=4, column=3, sticky=N+E+S+W)
        self.hand_visualizer_frame.rowconfigure("all", weight=1)

        self.visualizer_revealed_frame = Frame(self.hand_visualizer_frame)
        self.visualizer_revealed_frame.grid(row=0, column=0, sticky=N+E+W)

        self.visualizer_concealed_frame = Frame(self.hand_visualizer_frame)
        self.visualizer_concealed_frame.grid(row=1, column=0, sticky=E+W)

        self.visualizer_dawn_tile_frame = Frame(self.hand_visualizer_frame)
        self.visualizer_dawn_tile_frame.grid(row=2, column=0, sticky=W+S+E)

        self.visualizer_revealed_set_label = Label(self.visualizer_revealed_frame, text="Declared Sets:")
        self.visualizer_revealed_set_label.pack(side=TOP, anchor="nw")

        self.visualizer_concealed_set_label = Label(self.visualizer_concealed_frame, text="Concealed Tiles:")
        self.visualizer_concealed_set_label.pack(side=TOP, anchor="nw")

        self.visualizer_drawn_tile_label = Label(self.visualizer_dawn_tile_frame, text="Drawn Tile:")
        self.visualizer_drawn_tile_label.pack(side=TOP, anchor="nw")

        self.visualizer_drawn_tile_picture = Label(self.visualizer_dawn_tile_frame)

        for i in range(16):
            self.visualizer_concealed_set_tile_pictures += [Label(self.visualizer_concealed_frame)]
            self.visualizer_revealed_set_tile_pictures += [Label(self.visualizer_revealed_frame)]

    # TODO: Remove this
    def _debug_func(self):
        self.concealed_kong_entries[0].insert(0, "drg")
        self.concealed_kong_entries[1].insert(0, "")
        self.concealed_kong_entries[2].insert(0, "")
        self.concealed_kong_entries[3].insert(0, "")

        self.revealed_kong_entries[0].insert(0, "")
        self.revealed_kong_entries[1].insert(0, "")
        self.revealed_kong_entries[2].insert(0, "")
        self.revealed_kong_entries[3].insert(0, "")

        self.concealed_other_entries[0].insert(0, "b2")
        self.concealed_other_entries[1].insert(0, "b3")
        self.concealed_other_entries[2].insert(0, "b4")
        self.concealed_other_entries[3].insert(0, "b2")
        self.concealed_other_entries[4].insert(0, "b3")
        self.concealed_other_entries[5].insert(0, "b4")
        self.concealed_other_entries[6].insert(0, "b6")
        self.concealed_other_entries[7].insert(0, "")
        self.concealed_other_entries[8].insert(0, "")
        self.concealed_other_entries[9].insert(0, "")
        self.concealed_other_entries[10].insert(0, "")
        self.concealed_other_entries[11].insert(0, "")
        self.concealed_other_entries[12].insert(0, "")
        self.concealed_other_entries[13].insert(0, "")

        self.revealed_other_entries[0].insert(0, "b8")
        self.revealed_other_entries[1].insert(0, "b8")
        self.revealed_other_entries[2].insert(0, "b8")
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
        self.drawn_tile_entry.insert(0, "")

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
        self.final_tile_labelframe = None
        self.final_tile_entry = None
        self.final_tile_entry_cv = None
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
        self.visualizer_final_tile_label = None
        self.visualizer_revealed_set_tile_pictures = []
        self.visualizer_concealed_set_tile_pictures = []
        self.visualizer_final_tile_picture = None
        self.hand_entry_warning_label = None
        self.final_tile_drawn_or_discard_checkbutton_cv = None
        self.final_tile_drawn_or_discard_checkbutton = None
        self.last_tile_checkbutton_cv = None
        self.last_tile_checkbutton = None
        self.last_of_its_kind_checkbutton_cv = None
        self.last_of_its_kind_checkbutton = None
        self.replacement_tile_checkbutton_cv = None
        self.replacement_tile_checkbutton = None
        self.kong_rob_checkbutton_cv = None
        self.kong_rob_checkbutton = None
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
        final = self.final_tile_entry_cv.get()
        self_drawn_final = self.final_tile_drawn_or_discard_checkbutton_cv.get()
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
        self.calculator.set_special_conditions(self.last_tile_checkbutton_cv.get(),
                                               self.last_of_its_kind_checkbutton_cv.get(),
                                               self.replacement_tile_checkbutton_cv.get(),
                                               self.kong_rob_checkbutton_cv.get())
        self.calculator.set_hand(concealed, revealed, final, self_drawn_final, concealed_kongs, revealed_kongs,
                                 self.round_wind_cv.get(), self.seat_wind_cv.get())
        self._update_visualizer()

    def _update_visualizer(self):
        for l in self.visualizer_revealed_set_tile_pictures:
            l.pack_forget()
        for l in self.visualizer_concealed_set_tile_pictures:
            l.pack_forget()
        self.visualizer_final_tile_picture.pack_forget()

        concealed = self.calculator.pwh.concealed_tiles[:16]
        declared_kongs = self.calculator.pwh.declared_concealed_kongs[:4]
        c, p, k = self.calculator.pwh.get_revealed_sets()
        final = self.calculator.pwh.final_tile
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

        if final:
            self.visualizer_final_tile_picture.configure(image=final.ph)
            self.visualizer_final_tile_picture.pack(expand=YES, fill=BOTH)

    def _clear_hand_entry(self):
        for p in self.visualizer_concealed_set_tile_pictures:
            p.pack_forget()
        for p in self.visualizer_revealed_set_tile_pictures:
            p.pack_forget()
        self.visualizer_final_tile_picture.pack_forget()

        for e in self.concealed_other_entries:
            e.delete(0, 3)
        for e in self.concealed_kong_entries:
            e.delete(0, 3)
        for e in self.revealed_other_entries:
            e.delete(0, 3)
        for e in self.revealed_kong_entries:
            e.delete(0, 3)
        self.final_tile_entry.delete(0, 3)
        self.final_tile_drawn_or_discard_checkbutton_cv.set(0)
        self.last_tile_checkbutton_cv.set(0)
        self.last_of_its_kind_checkbutton_cv.set(0)
        self.replacement_tile_checkbutton_cv.set(0)
        self.kong_rob_checkbutton_cv.set(0)

    def create_hand_entry(self):
        self.entry_validation = self.register(self._check_valid_hand_entry)
        self.hand_entry_frame.rowconfigure("all", weight=1)
        self.hand_entry_frame.columnconfigure(0, weight=1)
        self.hand_entry_frame.columnconfigure(1, weight=1)
        self.hand_entry_frame.columnconfigure(2, weight=3)
        self.hand_entry_frame.columnconfigure(3, weight=5)
        instructions = "Instructions:\n" \
                       "Enter your hand, one tile per box\n" \
                       "(Excluding kongs. Type the tile once only.)\n" \
                       "If you have a kong you have not declared,\n" \
                       "enter it in 'others' instead of kong\n" \
                       "==========================================\n" \
                       "If you steal a tile, move the set to revealed sets\n" \
                       "If you win by stealing, set your final tile to non-self-drawn and move the other\n" \
                       "tiles to revealed\n" \
                       "===========================================\n" \
                       "b1-9 = Bamboo. Ex: b1 b5\n" \
                       "c1-9 = Characters. Ex c4 c8\n" \
                       "d1-9 = Dots. Ex d5 d6\n" \
                       "drw, drr, drg = White/Red/Green Dragon\n" \
                       "wn, we, ws, ww = North/East/South/West Wind\n"
        self.instructions_text = Message(self.hand_entry_frame, text=instructions, aspect=400)
        self.instructions_text.grid(row=0, rowspan=3, column=0, sticky=W+N+S)

        self.final_tile_labelframe = Labelframe(self.hand_entry_frame, text="Final Tile")
        self.final_tile_labelframe.grid(row=0, rowspan=3, column=1, padx=15, pady=2, sticky=N+S)
        self.final_tile_entry_cv = StringVar()
        self.final_tile_entry = Entry(self.final_tile_labelframe, validate="key", width=4,
                                      textvariable=self.final_tile_entry_cv,
                                      validatecommand=(self.entry_validation, '%P'))
        self.final_tile_entry.pack(side=TOP, expand=YES, anchor="s")

        self.final_tile_drawn_or_discard_checkbutton_cv = IntVar()
        self.final_tile_drawn_or_discard_checkbutton = Checkbutton(self.final_tile_labelframe, text="Self Drawn?",
                                                                   variable=
                                                                   self.final_tile_drawn_or_discard_checkbutton_cv)
        self.final_tile_drawn_or_discard_checkbutton.pack(side=TOP, anchor="w")
        self.final_tile_drawn_or_discard_checkbutton.invoke()
        self.final_tile_drawn_or_discard_checkbutton.invoke()
        self.final_tile_drawn_or_discard_checkbutton.configure(command=self._hand_change)

        self.last_tile_checkbutton_cv = IntVar()
        self.last_tile_checkbutton = Checkbutton(self.final_tile_labelframe, text="Last Tile in Game?",
                                                 variable=self.last_tile_checkbutton_cv)
        self.last_tile_checkbutton.pack(side=TOP, anchor="w")
        self.last_tile_checkbutton.invoke()
        self.last_tile_checkbutton.invoke()
        self.last_tile_checkbutton.configure(command=self._hand_change)

        self.last_of_its_kind_checkbutton_cv = IntVar()
        self.last_of_its_kind_checkbutton = Checkbutton(self.final_tile_labelframe, text="Last tile of its kind?",
                                                        variable=self.last_of_its_kind_checkbutton_cv)
        self.last_of_its_kind_checkbutton.pack(side=TOP, anchor="w")
        self.last_of_its_kind_checkbutton.invoke()
        self.last_of_its_kind_checkbutton.invoke()
        self.last_of_its_kind_checkbutton.configure(command=self._hand_change)

        self.replacement_tile_checkbutton_cv = IntVar()
        self.replacement_tile_checkbutton = Checkbutton(self.final_tile_labelframe, text="Replacement Tile?",
                                                        variable=self.replacement_tile_checkbutton_cv)
        self.replacement_tile_checkbutton.pack(side=TOP, anchor="w")
        self.replacement_tile_checkbutton.invoke()
        self.replacement_tile_checkbutton.invoke()
        self.replacement_tile_checkbutton.configure(command=self._hand_change)

        self.kong_rob_checkbutton_cv = IntVar()
        self.kong_rob_checkbutton = Checkbutton(self.final_tile_labelframe, text="Self Drawn?",
                                                variable=self.kong_rob_checkbutton_cv)
        self.kong_rob_checkbutton.pack(side=TOP, anchor="w")
        self.kong_rob_checkbutton.invoke()
        self.kong_rob_checkbutton.invoke()
        self.kong_rob_checkbutton.configure(command=self._hand_change)

        self.concealed_entry_labelframe = LabelFrame(self.hand_entry_frame, text="Concealed Tiles")
        self.concealed_entry_labelframe.grid(row=0, column=2, pady=8, sticky=N+S+W)
        self.revealed_entry_labelframe = Labelframe(self.hand_entry_frame, text="Revealed Tiles")
        self.revealed_entry_labelframe.grid(row=1, column=2, pady=8, sticky=N+S+W)

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

        warning_text = "IMPORTANT: Enter revealed tile sets specifically." \
                       "EXAMPLE: if you have 111222333 it does matter if you type it as\n" \
                       "111 222 333 VS. 123 123 123\n" \
                       " Also note that your tiles may" \
                       " disappear while you are typing. This is normal.\n" \
                       "Try not to mix up the order or put it in weird. :)"
        self.hand_entry_warning_label = Message(self.hand_entry_frame, text=warning_text, aspect=1200)
        self.hand_entry_warning_label.grid(row=2, column=2, sticky=W)

        self.clear_hand_button = Button(self.hand_entry_frame, text="Clear Hand", command=self._clear_hand_entry)
        self.clear_hand_button.grid(row=3, column=2, sticky=W+E)

        self.round_seat_wind_frame = Frame(self.hand_entry_frame)
        self.round_seat_wind_frame.grid(row=0, rowspan=3, column=3, padx=15, sticky=W+N+S)
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
        self.hand_visualizer_frame.grid(row=0, rowspan=3, padx=15, pady=4, column=4, sticky=N+E+S+W)
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

        self.visualizer_final_tile_label = Label(self.visualizer_dawn_tile_frame, text="Final Tile:")
        self.visualizer_final_tile_label.pack(side=TOP, anchor="nw")

        self.visualizer_final_tile_picture = Label(self.visualizer_dawn_tile_frame)

        for i in range(16):
            self.visualizer_concealed_set_tile_pictures += [Label(self.visualizer_concealed_frame)]
            self.visualizer_revealed_set_tile_pictures += [Label(self.visualizer_revealed_frame)]

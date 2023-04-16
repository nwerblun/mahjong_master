from tkinter import *
from tkinter.ttk import *  # Automatically replace some widgets with better versions
from hands import MahjongHands
from utilities import *
from score_calculator import Calculator
from game import VoidTile
from pathfinding import *
from functools import partial


class HandAssister(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master.rowconfigure("all", weight=1)
        self.hand_entry_frame = Frame(self.master, borderwidth=3, relief=GROOVE)
        self.hand_entry_frame.pack(expand=NO, side=TOP, fill=X)

        self.solutions_frame = Frame(self.master, borderwidth=3, relief=GROOVE)
        self.solutions_frame.pack(expand=YES, side=BOTTOM, fill=BOTH)
        self.solutions_frame.rowconfigure(0, weight=1)
        self.solutions_frame.columnconfigure(0, weight=1)

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
        self.visualizer_total_score_label = None
        self.visualizer_total_score_label_cv = None
        self.visualizer_total_score_label_style = None

        self.popup = None
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

        self.pathfinder = Pathfinder(self.calculator)
        self.pathfinder_pipe = None
        self.pathfinder_process = None
        self.pathfinder_poll_callback_id = None

        self.solutions_style = Style()
        self.solutions_style.configure("Solutions.TLabel",
                                       font=('Segoe UI', 12, "bold", "underline"), foreground="blue")
        self.solutions_label = Label(self.solutions_frame, anchor="center")
        self.solutions_label.grid(row=0, column=0, sticky=N+E+S+W)
        self.num_sols_to_find = 6
        self.solution_entries = []

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

    def _combobox_change(self, index, value, op):
        # Parameters are required by tkinter, but not needed
        self._hand_change()

    def _poll_status(self):
        self.solutions_label.configure(text="Loading...")
        self.solutions_label.grid(row=0, column=0, sticky=N+E+S+W)
        if self.pathfinder_process is None or self.pathfinder_pipe is None:
            return
        if not self.pathfinder_pipe.poll():
            self.pathfinder_poll_callback_id = self.after(10, self._poll_status)
            return
        final = self.pathfinder_pipe.recv()
        self.pathfinder_pipe.close()
        self.pathfinder_process.join()
        if len(final) == 0 or final is None:
            self.solutions_label.configure(text="Number of iterations with no solution exceeded.", anchor="center")
        self._update_solutions_area(final)

    def _update_solutions_area(self, final):
        for entr in self.solution_entries:
            entr.grid_remove()
            recursive_destroy(entr)
        self.solution_entries = []

        if final is None or len(final) == 0:
            return

        self.solutions_label.configure(text="Closest Solution(s) Found!\n", anchor="center")
        self.solutions_label.grid(row=0, column=0, columnspan=self.num_sols_to_find // 3, sticky=N)
        for i in range(len(final)):
            final_calc = final[i]
            if not final_calc.pwh:
                final_calc.pwh = PossibleWinningHand(final_calc.hand)
            val, breakdown = final_calc.get_score_summary()
            entr_frm = self._create_solution_entry(final_calc.hand, val, breakdown, i+1)
            col_to_use = i//3
            entr_frm.grid(row=i % 3, column=col_to_use, sticky=W)
            self.solution_entries += [entr_frm]
        self.solutions_frame.rowconfigure("all", weight=1)
        self.solutions_frame.columnconfigure("all", weight=1)

    def _create_solution_entry(self, hand, val, breakdown, sol_num):
        frm = Frame(self.solutions_frame)
        rvld = hand.revealed_tiles
        cld = hand.concealed_tiles
        ck = hand.declared_concealed_kongs
        entry_lbl = Label(frm, style="Solutions.TLabel", cursor="hand2",
                          text=str(sol_num) + ". Est. " + str(val) + " Points. Click for Breakdown")
        entry_lbl.grid(row=0, sticky=E+W+N)
        entry_lbl.bind("<ButtonRelease>", self._gen_solution_callback(breakdown))

        declared_frm = Frame(frm)
        declared_frm.grid(row=1, sticky=E+W)

        concealed_frm = Frame(frm)
        concealed_frm.grid(row=2, sticky=E+W)

        all_tiles = cld + rvld + ck
        all_old_tiles = self.calculator.hand.concealed_tiles + self.calculator.hand.revealed_tiles +\
            self.calculator.hand.declared_concealed_kongs
        uniques_new = list(set(all_tiles))
        new_tiles = [i for i in uniques_new if all_tiles.count(i) > all_old_tiles.count(i)]
        special_style = Style()
        special_style.configure("NewTile.TFrame", background="red")

        for t in cld:
            ph = t.gen_img()
            if t in new_tiles:
                x = Frame(concealed_frm, style="NewTile.TFrame", borderwidth=5, relief=GROOVE)
                x.pack(side=LEFT, fill=Y)
                x = Label(x, image=ph)
                new_tiles.pop(new_tiles.index(t))
            else:
                x = Label(concealed_frm, image=ph)
            x.ph = ph  # Avoid garbage collection
            x.pack(side=LEFT, fill=Y)

        for t in rvld:
            ph = t.gen_img()
            if t in new_tiles:
                x = Frame(declared_frm, style="NewTile.TFrame", borderwidth=5, relief=GROOVE)
                x.pack(side=LEFT, fill=Y)
                x = Label(x, image=ph)
                new_tiles.pop(new_tiles.index(t))
            else:
                x = Label(declared_frm, image=ph)
            x.ph = ph  # Avoid garbage collection
            x.pack(side=LEFT, fill=Y)

        if len(ck) >= 4:
            for i in range(0, len(ck), 4):
                ph = self.void_tile.gen_img()
                if ck[i] in new_tiles:
                    x = Frame(declared_frm, style="NewTile.TFrame", borderwidth=5, relief=GROOVE)
                    x.pack(side=LEFT, fill=Y)
                    x = Label(x, image=ph)
                    new_tiles.pop(new_tiles.index(t))
                else:
                    x = Label(concealed_frm, image=ph)
                x.ph = ph  # Avoid garbage collection
                x.pack(side=LEFT, fill=Y)

                ph = ck[i].gen_img()
                if ck[i] in new_tiles:
                    x = Frame(declared_frm, style="NewTile.TFrame", borderwidth=5, relief=GROOVE)
                    x.pack(side=LEFT, fill=Y)
                    x = Label(x, image=ph)
                    new_tiles.pop(new_tiles.index(t))
                else:
                    x = Label(concealed_frm, image=ph)
                x.ph = ph  # Avoid garbage collection
                x.pack(side=LEFT, fill=Y)

                if ck[i] in new_tiles:
                    x = Frame(declared_frm, style="NewTile.TFrame", borderwidth=5, relief=GROOVE)
                    x.pack(side=LEFT, fill=Y)
                    x = Label(x, image=ph)
                    new_tiles.pop(new_tiles.index(t))
                else:
                    x = Label(concealed_frm, image=ph)
                x.ph = ph  # Avoid garbage collection
                x.pack(side=LEFT, fill=Y)

                ph = self.void_tile.gen_img()
                if ck[i] in new_tiles:
                    x = Frame(declared_frm, style="NewTile.TFrame", borderwidth=5, relief=GROOVE)
                    x.pack(side=LEFT, fill=Y)
                    x = Label(x, image=ph)
                    new_tiles.pop(new_tiles.index(t))
                else:
                    x = Label(concealed_frm, image=ph)
                x.ph = ph  # Avoid garbage collection
                x.pack(side=LEFT, fill=Y)
        return frm

    def _gen_solution_callback(self, breakdown):
        def f(str_to_print, event):
            if self.popup:
                self.popup.destroy()
                self.popup = None
            self.popup = Toplevel()
            self.popup.title("Score Breakdown")
            self.popup.geometry("600x400")
            self.popup.resizable(height=False, width=False)
            frm = Frame(self.popup)
            frm.pack()
            lbl = Label(frm, text="Score Breakdown", justify=CENTER)
            lbl.grid(row=0, column=0, columnspan=4, sticky=N)
            if str_to_print == "":
                lbl = Label(frm, text="I don't know why, but there's no breakdown here.")
                lbl.grid(row=1, column=0, columnspan=4, sticky=N+E+W)
            else:
                for row, line in enumerate(str_to_print.split("\n")):
                    split_line = line.split(",")
                    lbl = Label(frm, text=split_line[0])
                    lbl.grid(row=row + 1, column=0)
                    lbl = Label(frm, text=split_line[1])
                    lbl.grid(row=row + 1, column=1)
                    lbl = Label(frm, text=split_line[2])
                    lbl.grid(row=row + 1, column=2)
                    lbl = Label(frm, text=split_line[3])
                    lbl.grid(row=row + 1, column=3)
        func = partial(f, breakdown)
        return func

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
        self.pathfinder = Pathfinder(self.calculator)
        self._update_visualizer()

        if self.pathfinder_process:
            # Kill it
            self.pathfinder_pipe.close()
            self.pathfinder_process.terminate()
            self.pathfinder_process.join()
            self.pathfinder_pipe = None
            self.pathfinder_process = None
        if self.pathfinder_poll_callback_id is not None:
            self.after_cancel(self.pathfinder_poll_callback_id)

        if self.pathfinder.ready_to_check():
            self.pathfinder_pipe, self.pathfinder_process = self.pathfinder.get_n_fastest_wins(n=self.num_sols_to_find)
            self.pathfinder_poll_callback_id = self.after(10, self._poll_status)
        else:
            self.solutions_label.configure(text="Please enter a valid starting point above.")
            self.solutions_label.grid(row=0, column=0, sticky=N+E+S+W)
            self._update_solutions_area(None)

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
            ph = t.gen_img()
            self.visualizer_concealed_set_tile_pictures[i].configure(image=ph)
            self.visualizer_concealed_set_tile_pictures[i].ph = ph  # Avoid GC on the img
            self.visualizer_concealed_set_tile_pictures[i].pack(side=LEFT, anchor="w")

        lbl_ctr = 0
        for r_set in c+p+k:
            if lbl_ctr + len(r_set) <= 16:
                for i in range(len(r_set) - 1):
                    ph = r_set[i].gen_img()
                    self.visualizer_revealed_set_tile_pictures[lbl_ctr].configure(image=ph)
                    self.visualizer_revealed_set_tile_pictures[lbl_ctr].ph = ph
                    self.visualizer_revealed_set_tile_pictures[lbl_ctr].pack(side=LEFT, anchor="w")
                    lbl_ctr += 1
                if len(r_set) > 1:
                    ph = r_set[-1].gen_img()
                    self.visualizer_revealed_set_tile_pictures[lbl_ctr].configure(image=ph)
                    self.visualizer_revealed_set_tile_pictures[lbl_ctr].ph = ph
                    self.visualizer_revealed_set_tile_pictures[lbl_ctr].pack(side=LEFT, padx=(0, 18), anchor="w")
                    lbl_ctr += 1

        start = False
        for d_set in declared_kongs:
            if lbl_ctr + len(d_set) <= 16:
                for i in range(len(d_set) - 1):
                    if not start:
                        ph = self.void_tile.gen_img()
                        self.visualizer_revealed_set_tile_pictures[lbl_ctr].configure(image=ph)
                        self.visualizer_revealed_set_tile_pictures[lbl_ctr].ph = ph
                        self.visualizer_revealed_set_tile_pictures[lbl_ctr].pack(side=LEFT, anchor="w")
                        start = True
                    else:
                        ph = d_set[i].gen_img()
                        self.visualizer_revealed_set_tile_pictures[lbl_ctr].configure(image=ph)
                        self.visualizer_revealed_set_tile_pictures[lbl_ctr].ph = ph
                        self.visualizer_revealed_set_tile_pictures[lbl_ctr].pack(side=LEFT, anchor="w")
                    lbl_ctr += 1
                if len(d_set) > 1:
                    ph = self.void_tile.gen_img()
                    self.visualizer_revealed_set_tile_pictures[lbl_ctr].configure(image=ph)
                    self.visualizer_revealed_set_tile_pictures[lbl_ctr].ph = ph
                    self.visualizer_revealed_set_tile_pictures[lbl_ctr].pack(side=LEFT, padx=(0, 18), anchor="w")
                    lbl_ctr += 1
                start = False

        if final:
            ph = final.gen_img()
            self.visualizer_final_tile_picture.configure(image=ph)
            self.visualizer_final_tile_picture.ph = ph
            self.visualizer_final_tile_picture.pack(expand=YES, fill=BOTH)
        score = self.calculator.total_hand_value
        if score >= 8:
            self.visualizer_total_score_label_style.configure("TotalPoints.TLabel", foreground="green")
        else:
            self.visualizer_total_score_label_style.configure("TotalPoints.TLabel", foreground="red")
        self.visualizer_total_score_label_cv.set("Total Score (click for info): "+str(score))

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
        self.round_wind_cv.set("East")
        self.seat_wind_cv.set("East")
        self.last_tile_checkbutton_cv.set(0)
        self.last_of_its_kind_checkbutton_cv.set(0)
        self.replacement_tile_checkbutton_cv.set(0)
        self.kong_rob_checkbutton_cv.set(0)

    def _create_score_popup(self, event):
        if self.popup:
            self.popup.destroy()
            self.popup = None
        self.popup = Toplevel()
        self.popup.title("Score Breakdown")
        self.popup.geometry("600x400")
        self.popup.resizable(height=False, width=False)
        frm = Frame(self.popup)
        frm.pack()
        breakdown = self.calculator.score_breakdown
        lbl = Label(frm, text="Score Breakdown", justify=CENTER)
        lbl.grid(row=0, column=0, columnspan=4, sticky=N)
        if breakdown == "":
            lbl = Label(frm, text="Invalid hand. Enter a valid hand to see the score.")
            lbl.grid(row=1, column=0, columnspan=4, sticky=N+E+W)
        else:
            for row, line in enumerate(breakdown.split("\n")):
                split_line = line.split(",")
                lbl = Label(frm, text=split_line[0])
                lbl.grid(row=row+1, column=0)
                lbl = Label(frm, text=split_line[1])
                lbl.grid(row=row+1, column=1)
                lbl = Label(frm, text=split_line[2])
                lbl.grid(row=row+1, column=2)
                lbl = Label(frm, text=split_line[3])
                lbl.grid(row=row+1, column=3)

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

        self.visualizer_total_score_label_cv = StringVar()
        self.visualizer_total_score_label_style = Style()
        self.visualizer_total_score_label_style.configure("TotalPoints.TLabel",
                                                          font=('Segoe UI', 14, "bold"), foreground="red")

        self.visualizer_total_score_label = Label(self.hand_visualizer_frame,
                                                  textvariable=self.visualizer_total_score_label_cv,
                                                  style="TotalPoints.TLabel",
                                                  cursor="hand2")
        self.visualizer_total_score_label_cv.set("Total Score (click for info): ")
        self.visualizer_total_score_label.grid(row=3, column=0, sticky=W+S)
        self.visualizer_total_score_label.bind("<ButtonRelease>", self._create_score_popup)

        self.round_wind_cv.trace("w", self._combobox_change)
        self.seat_wind_cv.trace("w", self._combobox_change)


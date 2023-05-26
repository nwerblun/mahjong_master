from tkinter import *
from tkinter.ttk import *  # Automatically replace some widgets with better versions
from PIL import ImageTk, ImageGrab
from hands import MahjongHands
from utilities import *
from score_calculator import Calculator
from game import *
from pathfinding import *
import pyautogui as pa
import win32gui
# Ignore these errors. The top level application adds the correct path to sys.path for imports
from cv2 import resize as img_resize
import numpy as np
from yolo_model import predict_on_img_obj, load_model
from input_output_utils import get_pred_output_img
import yolo_globals as yg


class HandAnalyzer(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master.rowconfigure("all", weight=1)
        self.calculator = Calculator()
        self.pathfinder = Pathfinder(self.calculator)
        self.yolo = load_model(True)
        self.app_select_frame = Frame(self.master, borderwidth=3, relief=GROOVE)
        self.app_select_frame.pack(expand=NO, side=TOP, fill=X)

        self.app_preview_frame = Frame(self.master, borderwidth=3, relief=GROOVE)
        self.app_preview_frame.pack(expand=NO, side=TOP, fill=X)
        self.auto_hand_visualizer_frame = None
        self.preview_label = None
        self.preview_callback_id = None
        self.preview_img_lbl = Label(self.app_preview_frame)
        self.xy, self.wh, self.conf, self.cls = None, None, None, None
        self.nms_prediction_last = None
        self.discarded_tiles = []
        self.popup = None
        self.discard_popup_label = None
        self.discard_popup_label_cv = None
        self.discard_popup_label_style = None

        self.auto_hand_concealed_tiles_frame = None
        self.auto_hand_declared_tiles_frame = None
        self.auto_hand_final_tile_frame = None
        self.visualizer_declared_set_tile_pictures = []
        self.visualizer_concealed_set_tile_pictures = []
        self.visualizer_final_tile_picture = None
        self.void_tile = VoidTile()
        self.round_wind_combobox_cv = None
        self.seat_wind_combobox_cv = None
        self.round_wind_combobox = None
        self.seat_wind_combobox = None

        self.app_select_combobox = None
        self.app_select_combobox_cv = None
        self.selected_window = -1

    def _update_active_apps_combobox_selection(self, event):
        all_windows = pa.getAllWindows()
        all_windows_titles = [e.title for e in all_windows
                              if e.title != "" and e.title != self.winfo_toplevel().title()]
        self.app_select_combobox.configure(values=all_windows_titles)

    def _active_apps_sel_to_hwnd(self, index, value, op):
        self.preview_img_lbl.configure(image=None)
        self.preview_img_lbl.ph = None

        if self.app_select_combobox_cv.get() == "":
            self.selected_window = -1
            return
        all_windows = pa.getAllWindows()
        all_windows_hwnd = [e._hWnd for e in all_windows if e.title != ""]
        all_windows_titles = [e.title for e in all_windows if e.title != ""]
        ind = all_windows_titles.index(self.app_select_combobox_cv.get())
        self.selected_window = all_windows_hwnd[ind]

    def _active_app_preview_loop(self):
        if win32gui.GetForegroundWindow() != self.selected_window:
            self.preview_label.configure(text="Waiting for a window to be selected "
                                              "or for chosen application to become active.")
            self.preview_callback_id = self.after(100, self._active_app_preview_loop)
            return
        else:
            self.preview_label.configure(text="Preview of Chosen Application")
            sc = pa.screenshot()
            img = np.array(sc)
            self.xy, self.wh, self.conf, self.cls = predict_on_img_obj(img, True, self.yolo)
            img_with_boxes, nms_pred = get_pred_output_img(img, self.xy, self.wh,
                                                           self.conf, self.cls,
                                                           class_thresh=0.95, conf_thresh=0.75,
                                                           nms_iou_thresh=0.2)
            self._prediction_to_calc_and_pf(nms_pred)
            new_h = int(self.winfo_toplevel().winfo_height()*0.5)
            new_w = int(new_h * 16 / 9)
            img = img_resize(img_with_boxes, (new_w, new_h))
            ph = ImageTk.PhotoImage(Image.fromarray(img))
            self.preview_img_lbl.configure(image=ph, anchor="center")
            self.preview_img_lbl.ph = ph  # Avoid garbage collection
        self.preview_callback_id = self.after(100, self._active_app_preview_loop)

    def _prediction_to_calc_and_pf(self, nms_pred):
        # All xywh in % of img size
        # Each entry contains: box_xywh, predicted class prob, predicted class name, predicted conf
        player_hand_xy_min = [0.21, 0.84]
        player_hand_xy_max = [0.756, 0.967]
        all_in_player_hand = [t for t in nms_pred if (player_hand_xy_min[0] <= t[0][0] <= player_hand_xy_max[0] and
                                                      player_hand_xy_min[1] <= t[0][1] <= player_hand_xy_max[1])]
        self.discarded_tiles = [t[2] for t in nms_pred if (player_hand_xy_min[0] > t[0][0] or
                                                           t[0][0] > player_hand_xy_max[0] and
                                                           player_hand_xy_min[1] > t[0][1] or
                                                           t[0][1] < player_hand_xy_max[1])]
        all_in_player_hand_last_pred = [t for t in self.nms_prediction_last if
                                        (player_hand_xy_min[0] <= t[0][0] <= player_hand_xy_max[0] and
                                         player_hand_xy_min[1] <= t[0][1] <= player_hand_xy_max[1])]

        all_in_hand_names = [t[2] for t in all_in_player_hand]
        all_in_hand_last_names = [t[2] for t in all_in_player_hand_last_pred]
        difference = []
        difference_amts = []
        for _, _, tile_name, _ in all_in_player_hand:
            if all_in_hand_names.count(tile_name) != all_in_hand_last_names.count(tile_name) and\
                    tile_name not in difference:
                difference += [tile_name]
                difference_amts += [all_in_hand_names.count(tile_name) - all_in_hand_last_names.count(tile_name)]

        if len(difference) == 0:
            self.nms_prediction_last = nms_pred
            self.discarded_tiles = []
            return

        concealed, revealed = [], []
        concealed_kongs, revealed_kongs = [], []
        not_concealed = []
        final = None
        self_drawn_final = False

        # Split into definitely concealed and revealed, but not sure what type of revealed
        for box, _, tile_name, _ in all_in_player_hand:
            box_wh = box[2:]
            box_area = box_wh[0] * box_wh[1]
            # Concealed tiles have an area of about 0.0035, revealed are around 0.002
            if box_area > 0.003:
                concealed += [tile_name]
            else:
                not_concealed += [[box, tile_name]]

        if len(concealed) == 0:
            print("Found no concealed tiles. Something went wrong with scanning, maybe?")
            self.nms_prediction_last = None
            self.discarded_tiles = []
            return

        index_of_final_tile = -1
        if self.nms_prediction_last is None:
            # Temporary measure until we get a second scan
            final = concealed[index_of_final_tile]
            final_box_area = 0

        # TODO: Remove the last drawn tile from concealed/revealed since it gets added in automatically in hand class
        # TODO: Handle removing the last tile if it was a kong
        else:
            # Drew/stole a tile, haven't discarded one yet
            if len(difference) == 1:
                pass
            # Drew a tile, discarded a different tile you have >1 of or kong + replacement tile
            elif len(difference) == 2:
                pass
            else:
                print("More than 3 differences between current and last hand, bailing out")
                self.nms_prediction_last = nms_pred
                self.discarded_tiles = []
                return

        # Check if tile size indicates concealed/not concealed
        self_drawn_final = final_box_area > 0.003

        # Sort by x value of the box center
        not_concealed = sorted(not_concealed, key=lambda x: x[0][0])
        counter = 0
        while counter < 10 and len(not_concealed) > 0:
            if len(not_concealed) == 1:
                concealed_kongs += [not_concealed[0][1]]
                # Shortcut to end the while loop. I didn't use break because whatever
                not_concealed = []
            # After concealed kongs, must always be 3's or 4's. If not then there's a mistake.
            elif len(not_concealed) < 3:
                print("Something is probably wrong with detection/splitting to not concealed")
                print(not_concealed)
                raise ValueError("Cannot have fewer than 3 tiles in revealed section")
            # Only one set, just put it in revealed
            elif len(not_concealed) == 3:
                for _, tile_name in not_concealed:
                    revealed += [tile_name]
                not_concealed = []
            elif len(not_concealed) >= 4:
                # Kong is about 0.06 apart, horz. tiles are about 0.04 apart, vert. about 0.03 apart
                one_far_from_two = not_concealed[1][0][0] - not_concealed[0][0][0] > 0.05
                four_far_from_three = not_concealed[3][0][0] - not_concealed[2][0][0] > 0.05
                t0, t1, t2 = Tile(not_concealed[0][1]), Tile(not_concealed[1][1]), Tile(not_concealed[2][1])
                t3 = Tile(not_concealed[3][1])
                # Chows can sometimes be out of order when stolen
                first_three_in_order = sorted([t0, t1, t2])
                first_three_sequential = first_three_in_order[1].is_sequential_to(first_three_in_order[0]) and\
                    first_three_in_order[2].is_sequential_to(first_three_in_order[1])
                first_three_equal_not_fourth = t0 == t1 and t1 == t2 and t3 != t2

                if len(not_concealed) > 4:
                    t4 = Tile(not_concealed[4][1])
                    first_four_equal_fifth_not_sequential = (t0 == t1 and t1 == t2 and t2 == t3)\
                        and not t4.is_sequential_to(t3)
                    first_four_equal_fifth_sequential = (t0 == t1 and t1 == t2 and t2 == t3)\
                        and t4.is_sequential_to(t3)

                # 1 is far from 2 (concealed kong), pop off first tile only
                if one_far_from_two:
                    concealed_kongs += [not_concealed.pop(0)[1]]
                # 4 is far from 3 (concealed kong, first 3 must be some kind of set) pop first 4 into respective sets
                elif four_far_from_three:
                    for _ in range(3):
                        revealed += [not_concealed.pop(0)[1]]
                    concealed_kongs += [not_concealed.pop(0)[1]]
                # 1 -> 2 -> 3 (chow), pop off first 3 only OR 1==2==3!=4 (pung), pop off first 3 only
                elif first_three_sequential or first_three_equal_not_fourth:
                    for _ in range(3):
                        revealed += [not_concealed.pop(0)[1]]
                # 1==2==3==4 and 4 !-> 5 (kong), pop off first 4
                elif len(not_concealed) > 4 and first_four_equal_fifth_not_sequential:
                    revealed_kongs += [not_concealed.pop(0)[1]]
                    for _ in range(3):
                        not_concealed.pop(0)
                # 1==2==3==4 and 4->5, maybe kong maybe pung
                # Kong case, 3/4 are close together
                elif len(not_concealed) > 4 and first_four_equal_fifth_sequential and not four_far_from_three:
                    revealed_kongs += [not_concealed.pop(0)[1]]
                    for _ in range(3):
                        not_concealed.pop(0)
                # pung + chow
                elif len(not_concealed) > 4 and first_four_equal_fifth_sequential and four_far_from_three:
                    for _ in range(3):
                        revealed += [not_concealed.pop(0)[1]]
                else:
                    print("Something is probably wrong with these tiles")
                    print(not_concealed)
        if counter == 10:
            print("Reached max iterations in trying to decode detected hand")

        self.calculator.set_hand(concealed, revealed, final, self_drawn_final, concealed_kongs, revealed_kongs,
                                 self.round_wind_combobox_cv.get(), self.seat_wind_combobox_cv.get())
        self.pathfinder = Pathfinder(self.calculator)
        self.nms_prediction_last = nms_pred
        self._update_detected_hand()

    def _update_detected_hand(self):
        for l in self.visualizer_declared_set_tile_pictures:
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
        for r_set in c + p + k:
            if lbl_ctr + len(r_set) <= 16:
                for i in range(len(r_set) - 1):
                    ph = r_set[i].gen_img()
                    self.visualizer_declared_set_tile_pictures[lbl_ctr].configure(image=ph)
                    self.visualizer_declared_set_tile_pictures[lbl_ctr].ph = ph
                    self.visualizer_declared_set_tile_pictures[lbl_ctr].pack(side=LEFT, anchor="w")
                    lbl_ctr += 1
                if len(r_set) > 1:
                    ph = r_set[-1].gen_img()
                    self.visualizer_declared_set_tile_pictures[lbl_ctr].configure(image=ph)
                    self.visualizer_declared_set_tile_pictures[lbl_ctr].ph = ph
                    self.visualizer_declared_set_tile_pictures[lbl_ctr].pack(side=LEFT, padx=(0, 18), anchor="w")
                    lbl_ctr += 1

        start = False
        for d_set in declared_kongs:
            if lbl_ctr + len(d_set) <= 16:
                for i in range(len(d_set) - 1):
                    if not start:
                        ph = self.void_tile.gen_img()
                        self.visualizer_declared_set_tile_pictures[lbl_ctr].configure(image=ph)
                        self.visualizer_declared_set_tile_pictures[lbl_ctr].ph = ph
                        self.visualizer_declared_set_tile_pictures[lbl_ctr].pack(side=LEFT, anchor="w")
                        start = True
                    else:
                        ph = d_set[i].gen_img()
                        self.visualizer_declared_set_tile_pictures[lbl_ctr].configure(image=ph)
                        self.visualizer_declared_set_tile_pictures[lbl_ctr].ph = ph
                        self.visualizer_declared_set_tile_pictures[lbl_ctr].pack(side=LEFT, anchor="w")
                    lbl_ctr += 1
                if len(d_set) > 1:
                    ph = self.void_tile.gen_img()
                    self.visualizer_declared_set_tile_pictures[lbl_ctr].configure(image=ph)
                    self.visualizer_declared_set_tile_pictures[lbl_ctr].ph = ph
                    self.visualizer_declared_set_tile_pictures[lbl_ctr].pack(side=LEFT, padx=(0, 18), anchor="w")
                    lbl_ctr += 1
                start = False

        if final:
            ph = final.gen_img()
            self.visualizer_final_tile_picture.configure(image=ph)
            self.visualizer_final_tile_picture.ph = ph
            self.visualizer_final_tile_picture.pack(expand=YES, fill=BOTH)

    def _create_discarded_tiles_popup(self, event):
        if self.popup:
            self.popup.destroy()
            self.popup = None
        self.popup = Toplevel()
        self.popup.title("Discarded Tiles")
        self.popup.geometry("400x900")
        self.popup.resizable(height=False, width=False)
        frm = Frame(self.popup)
        frm.pack()
        e = Label(frm, text="Currently Detected Discarded Tiles:")
        e.grid(row=0, column=0, columnspan=3)
        uniques = list(set(self.discarded_tiles))
        discard_counts = [(t, self.discarded_tiles.count(t)) for t in uniques]
        # Sort by tile name
        discard_counts = sorted(discard_counts, key=lambda x: x[0])
        for i in range(len(discard_counts)):
            tile_name, count = discard_counts[i]
            e = Label(frm, text=tile_name)
            e.grid(row=i+1, column=0)
            e = Label(frm, text="\t\t")
            e.grid(row=i+1, column=1)
            e = Label(frm, text="x"+str(count))
            e.grid(row=i+1, column=2)

    def create_application_selector(self):
        e = Label(self.app_select_frame, text="Select an Application to Monitor:")
        e.pack(fill=X, pady=4, anchor="w")
        self.app_select_combobox_cv = StringVar()
        self.app_select_combobox = Combobox(self.app_select_frame, textvariable=self.app_select_combobox_cv,
                                            exportselection=False, justify=LEFT)
        self.app_select_combobox.bind("<Button>", self._update_active_apps_combobox_selection)
        self.app_select_combobox.pack(expand=YES, fill=X, pady=4, anchor="center")
        self.app_select_combobox_cv.trace("w", self._active_apps_sel_to_hwnd)
        self.app_select_combobox.state(['!disabled', 'readonly'])
        self.app_select_combobox_cv.set("")

    def create_application_preview(self):
        self.preview_label = Label(self.app_preview_frame, text="Waiting for a window to be selected"
                                                                " or for chosen application to become active.")
        self.preview_label.pack(side=TOP, expand=NO, fill=X, anchor="center")
        self.preview_callback_id = self.after(100, self._active_app_preview_loop)
        self.preview_img_lbl.pack(side=LEFT, expand=YES, fill=BOTH, anchor="w")

        self.auto_hand_visualizer_frame = Frame(self.app_preview_frame)
        self.auto_hand_visualizer_frame.pack(expand=YES, side=RIGHT, fill=BOTH)

        e = Label(self.auto_hand_visualizer_frame, text="Detected Concealed Tiles:")
        e.grid(row=0, column=0, sticky=W)
        self.auto_hand_concealed_tiles_frame = Frame(self.auto_hand_visualizer_frame)
        self.auto_hand_concealed_tiles_frame.grid(row=1, column=0, sticky=W+E)

        e = Label(self.auto_hand_visualizer_frame, text="Detected Declared Tiles:")
        e.grid(row=2, column=0, sticky=W)
        self.auto_hand_declared_tiles_frame = Frame(self.auto_hand_visualizer_frame)
        self.auto_hand_declared_tiles_frame.grid(row=3, column=0, sticky=W+E)

        e = Label(self.auto_hand_visualizer_frame,
                  text="Last Drawn Tile (May be initially incorrect, may not detect every discarded tile):")
        e.grid(row=4, column=0, sticky=W)
        self.auto_hand_final_tile_frame = Frame(self.auto_hand_visualizer_frame)
        self.auto_hand_final_tile_frame.grid(row=5, column=0, sticky=W+E)

        self.visualizer_final_tile_picture = Label(self.auto_hand_final_tile_frame)

        for i in range(16):
            self.visualizer_concealed_set_tile_pictures += [Label(self.auto_hand_concealed_tiles_frame)]
            self.visualizer_declared_set_tile_pictures += [Label(self.auto_hand_declared_tiles_frame)]

        self.discard_popup_label_cv = StringVar()
        self.discard_popup_label_style = Style()
        self.discard_popup_label_style.configure("Discarded.TLabel", font=('Segoe UI', 12, "bold", "underline"), foreground="blue")
        self.discard_popup_label = Label(self.auto_hand_visualizer_frame, textvariable=self.discard_popup_label_cv,
                                         style="Discarded.TLabel", cursor="hand2")
        self.discard_popup_label_cv.set("Click for a list of recognized discarded tiles")
        self.discard_popup_label.grid(row=6, column=0, sticky=W)
        self.discard_popup_label.bind("<ButtonRelease>", self._create_discarded_tiles_popup)

        e = Label(self.auto_hand_visualizer_frame, text="Round Wind:")
        e.grid(row=1, column=1, sticky=W)

        self.round_wind_combobox_cv = StringVar()
        self.round_wind_combobox = Combobox(self.auto_hand_visualizer_frame, textvariable=self.round_wind_combobox_cv,
                                            exportselection=False, justify=LEFT,
                                            values=("North", "East", "South", "West"))
        self.round_wind_combobox.grid(row=2, column=1, sticky=W)
        self.round_wind_combobox.state(['!disabled', 'readonly'])
        self.round_wind_combobox_cv.set("East")

        e = Label(self.auto_hand_visualizer_frame, text="Seat Wind:")
        e.grid(row=3, column=1, sticky=W)
        self.seat_wind_combobox_cv = StringVar()
        self.seat_wind_combobox = Combobox(self.auto_hand_visualizer_frame, textvariable=self.seat_wind_combobox_cv,
                                           exportselection=False, justify=LEFT,
                                           values=("North", "East", "South", "West"))
        self.seat_wind_combobox.grid(row=4, column=1, sticky=W)
        self.seat_wind_combobox.state(['!disabled', 'readonly'])
        self.seat_wind_combobox_cv.set("East")

        self.auto_hand_visualizer_frame.rowconfigure("all", weight=1)
        self.auto_hand_visualizer_frame.columnconfigure("all", weight=1)

    def create_auto_hand_visualizer(self):
        if self.preview_label is None:
            self.create_application_preview()

    def create_solver_section(self):
        pass

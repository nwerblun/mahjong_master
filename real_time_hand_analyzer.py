from tkinter import *
from tkinter.ttk import *  # Automatically replace some widgets with better versions
from PIL import ImageTk, ImageGrab
from hands import MahjongHands
from utilities import *
from score_calculator import Calculator
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
        self.preview_label = None
        self.preview_callback_id = None
        self.preview_img_lbl = Label(self.app_preview_frame)
        self.xy, self.wh, self.conf, self.cls = None, None, None, None
        self.nms_prediction_last = None

        self.app_select_combobox = None
        self.app_select_combobox_cv = None
        self.selected_window = -1

        self.auto_hand_visualizer_frame = Frame(self.app_preview_frame)
        self.auto_hand_visualizer_frame.pack(expand=YES, side=RIGHT, fill=BOTH)

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
        # box_xywh, predicted class prob, predicted class, name predicted conf
        player_hand_xy_min = [0.21, 0.967]
        player_hand_xy_max = [0.756, 0.84]
        concealed, revealed, discarded = [], [], []
        concealed_kongs, revealed_kongs = [], []
        not_concealed = []
        final = None
        self_drawn_final = False

        # Split into definitely concealed and revealed, but not sure what type of revealed
        for box, _, tile_name, _ in nms_pred:
            box_xy = box[:2]
            box_wh = box[2:]
            if player_hand_xy_min[0] <= box_xy[0] <= player_hand_xy_max[0]:
                if player_hand_xy_min[1] <= box_xy[1] <= player_hand_xy_max[1]:
                    box_area = box_wh[0] * box_wh[1]
                    if box_area > 0.003:
                        concealed += [tile_name]
                    else:
                        not_concealed += [[box, tile_name]]

        if self.nms_prediction_last is None:
            # Temporary measure until we get a second scan
            final = concealed[-1]
        else:
            # TODO: calculate the difference in tiles and set it as last
            pass

        # TODO: Add an if statement that checks if the box of the last tile is big enough to be concealed/revealed

        # Sort by x value of the box center
        not_concealed = sorted(not_concealed, key=lambda x: x[0][0])
        while len(not_concealed) > 0:
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
                # 1==2==3 and 1 is close to 2 (chow)
                if False:
                    pass
                # 1 is far from 2, 2/3/4 are sequential (concealed kong + chow)
                elif False:
                    pass
                # 1 far from 2==3==4 (concealed kong + pung)
                elif False:
                    pass
                # 1 close to 2, 1==2==3!=4 (pung)
                elif False:
                    pass
                # First and second are close, 1==2==3==4 (revealed kong)
                elif False:
                    pass
                else:
                    print("Something is probably wrong with these tiles")
                    print(not_concealed)

        # TODO: Add wind dropdowns
        self.calculator.set_hand(concealed, revealed, final, self_drawn_final, concealed_kongs, revealed_kongs,
                                 "East", "East")
        self.pathfinder = Pathfinder(self.calculator)
        self.nms_prediction_last = nms_pred

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

    def create_auto_hand_visualizer(self):
        if self.preview_label is None:
            self.create_application_preview()

    def create_solver_section(self):
        pass

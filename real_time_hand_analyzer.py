from tkinter import *
from tkinter.ttk import *  # Automatically replace some widgets with better versions
from PIL import ImageTk, ImageGrab
from hands import MahjongHands
from utilities import *
from score_calculator import Calculator
from pathfinding import *
import pyautogui as pa
import win32gui
from cv2 import resize as img_resize
import numpy as np
from yolo_model import predict_on_img_obj, load_model
from input_output_utils import get_pred_output_img


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
        self.nms_prediction = None

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
            img_with_boxes, self.nms_prediction = get_pred_output_img(img, self.xy, self.wh,
                                                                      self.conf, self.cls,
                                                                      class_thresh=0.95, conf_thresh=0.75,
                                                                      nms_iou_thresh=0.2)
            new_h = int(self.winfo_toplevel().winfo_height()*0.5)
            new_w = int(new_h * 16 / 9)
            img = img_resize(img_with_boxes, (new_w, new_h))
            ph = ImageTk.PhotoImage(Image.fromarray(img))
            self.preview_img_lbl.configure(image=ph, anchor="center")
            self.preview_img_lbl.ph = ph  # Avoid garbage collection
        self.preview_callback_id = self.after(100, self._active_app_preview_loop)

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

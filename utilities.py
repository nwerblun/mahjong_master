from tkinter import *
from tkinter.ttk import *  # Automatically replace some widgets with better versions
import numpy as np


def recursive_destroy(root):
    for c in root.winfo_children():
        recursive_destroy(c)
    root.destroy()


def bind_all_children(parent, event_name, func):
    for c in parent.winfo_children():
        c.bind(event_name, func)
        bind_all_children(c, event_name, func)


def get_index_of(arr, search_key):
    x = np.flatnonzero(np.core.defchararray.find(arr, search_key) != -1)
    if not len(x):
        return -1
    return x[0]

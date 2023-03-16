from tkinter import *
from tkinter.ttk import *  # Automatically replace some widgets with better versions
from game import *
from pathfinding import *
from hands import MahjongHands

# Contains representations of a hand, computes sets, what hand conditions are met, and current total point value


class Calculator:
    # class calculator: has a hand, get_current_value() -> figure out how many points it's currently worth
    # ##############instantiates a pathfinder, get_top_n_winners() -> list of tuples of (Hand, points, {})
    # #######################k, v pairs of that {} are hand title: point value
    # Make a set of special hands for things like knitted seq?
    def __init__(self):
        self.hand = Hand()
        self.hand_titles = MahjongHands.get_hand_titles()
        self.official_point_values = MahjongHands.get_point_values()
        self.voids = MahjongHands.get_voids()

    def on_hand_change(self, concealed_tile_names, revealed_tile_names, drawn_tile, concealed_kongs, revealed_kongs):
        # Handle adding/removing tiles typed in by the user
        pass

    def _check_conditions(self):
        pass

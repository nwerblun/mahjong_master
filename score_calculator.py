from tkinter import *
from tkinter.ttk import *  # Automatically replace some widgets with better versions
from game import *
from pathfinding import *
from hands import MahjongHands

# Contains representations of a hand, computes sets, what hand conditions are met, and current total point value


class Calculator:
    def __init__(self):
        self.hand = Hand()
        self.hand_titles = MahjongHands.get_hand_titles()
        self.official_point_values = MahjongHands.get_point_values()
        self.voids = MahjongHands.get_voids()

    def set_hand(self, concealed_tile_names, revealed_tile_names, drawn_tile, declared_concealed_kongs, revealed_kongs):
        # Handle adding/removing tiles typed in by the user
        self.hand.clear_hand()
        for n in concealed_tile_names:
            self.hand.add_tile_to_hand(False, n, more=True)
        for n in revealed_tile_names:
            self.hand.add_tile_to_hand(True, n, more=True)
        for k in declared_concealed_kongs:
            self.hand.add_declared_concealed_kong_to_hand(k)
        for k in revealed_kongs:
            self.hand.add_revealed_kong_to_hand(k)
        self.hand.set_drawn_tile(drawn_tile)

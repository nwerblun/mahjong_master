from tkinter import *
from tkinter.ttk import *  # Automatically replace some widgets with better versions
from game import *
from pathfinding import *
from hands import MahjongHands
from scoring_conditions import *


class Calculator:
    def __init__(self):
        self.hand = Hand()
        self.pwh = None
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
        self.pwh = PossibleWinningHand(self.hand)
        self._score_winning_sets()

    def _check_special_cases(self):
        return self

    def _score_winning_sets(self):
        # Format (used=False, excluded=False)
        # if used to make a set, used=True.
        # If used with a set that has used=True, excluded = True.
        # If attempting to use with a set that is excluded, don't.
        # Go through conditions in reverse order so exclusionary rule skips cheap fans.
        for hand_dict in self.pwh.four_set_pair_hands:
            tilesets = {"kongs": [], "chows": [], "pungs": []}
            for s in hand_dict["declared_concealed_kongs"]:
                tilesets["kongs"] += [TileSet(s, "kong", concealed=True, declared=True)]
            for s in hand_dict["concealed_kongs"]:
                tilesets["kongs"] += [TileSet(s, "kong", concealed=True, declared=False)]
            for s in hand_dict["revealed_kongs"]:
                tilesets["kongs"] += [TileSet(s, "kong")]
            for s in hand_dict["concealed_chows"]:
                tilesets["chows"] += [TileSet(s, "chow", concealed=True)]
            for s in hand_dict["revealed_chows"]:
                tilesets["chows"] += [TileSet(s, "chow")]
            for s in hand_dict["concealed_pungs"]:
                tilesets["pungs"] += [TileSet(s, "pung", concealed=True)]
            for s in hand_dict["revealed_pungs"]:
                tilesets["pungs"] += [TileSet(s, "pung")]

            amt = mixed_double_chow(tilesets["chows"])
            hand_dict["point_conditions"][1] = amt
            amt = pure_double_chow(tilesets["chows"])
            hand_dict["point_conditions"][0] = amt

        # check special cases
        # after checking all hands
        # check chicken hand





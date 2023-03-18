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
        self._check_conditions()

    def _get_all_permutations(self, c_chows, c_pungs, c_kongs):
        # Important to note that declared concealed kongs are not considered here, because they are permanent.
        chow_perms = []
        pung_perms = []
        kong_perms = []
        extra_tile_perms = []
        for chow in c_chows:
            chow_perms += [chow]
            for pung in c_pungs:
                temp0 = list(set(chow + pung))
                if len(temp0) == len(pung) + len(chow):
                    for kong in c_kongs:
                        temp1 = list(set(chow + pung + kong))
                        temp2 = list(set(chow + kong))
                        if len(temp1) == len(chow) + len(pung) + len(kong):
                            pung_perms += [pung]
                            kong_perms += [kong]
                        else:
                            pung_perms += [pung]
                            kong_perms += [[]]
                        if len(temp2) == len(chow) + len(kong):
                            pass

        return 0

    def _check_conditions(self):
        if self.hand.get_num_tiles_in_hand() < 14 or self.hand.drawn_tile is None:
            return 0
        # Step 0: Assume hand is set and valid
        # Step 1: ask the hand for all possible pungs/chows from the current hand
        # Step 2: ask the hand for all current existing (revealed) sets and declared, concealed kongs
        # Step 3: generate all possible permutations of sets, making sure not to re-use tiles
        #    Step 3.5: Each permutation should subtract one concealed tile to replicate discarding
        # ???
        # Step 5: Somehow implement the non-repeat and exclusionary rule
        declared_concealed_kongs = self.hand.get_declared_concealed_kongs()
        potential_concealed_chows = self.hand.get_potential_concealed_chows()
        potential_concealed_pungs, potential_concealed_kongs = self.hand.get_potential_concealed_pungs_kongs()
        revealed_chows, revealed_pungs, revealed_kongs = self.hand.get_revealed_sets()
        chow_perms, pung_perms, kong_perms, extra_tile_perms = self._get_all_permutations(potential_concealed_chows,
                                                                                          potential_concealed_pungs,
                                                                                          potential_concealed_kongs)
        return

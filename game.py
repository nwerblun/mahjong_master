from tkinter import *
from tkinter.ttk import *  # Automatically replace some widgets with better versions
from hands import MahjongHands
from PIL import Image, ImageTk
from functools import total_ordering
from utilities import flatten_list
from copy import deepcopy
from itertools import permutations
# contains stuff that is useful in many files about the game.


@total_ordering
class Tile:
    valid_tile_names = [
        "b1",
        "b2",
        "b3",
        "b4",
        "b5",
        "b6",
        "b7",
        "b8",
        "b9",
        "c1",
        "c2",
        "c3",
        "c4",
        "c5",
        "c6",
        "c7",
        "c8",
        "c9",
        "d1",
        "d2",
        "d3",
        "d4",
        "d5",
        "d6",
        "d7",
        "d8",
        "d9",
        "drg",
        "drr",
        "drw",
        "we",
        "wn",
        "ws",
        "ww"
    ]

    def __init__(self, name):
        self.name = name
        self.picture_link = self.get_picture_link_from_name()
        self.img = Image.open(self.picture_link)
        self.ph = ImageTk.PhotoImage(self.img)
        if self.name[0] == "b":
            self.type = "bamboo"
        elif self.name[0] == "c":
            self.type = "character"
        elif self.name[0] == "d" and (not self.name[1] == "r"):
            self.type = "dot"
        elif self.name[:2] == "dr":
            self.type = "dragon"
        else:
            self.type = "wind"

    @staticmethod
    def is_valid_name(name):
        return name in Tile.valid_tile_names

    def get_picture_link_from_name(self):
        ind = MahjongHands.tile_names.index(self.name)
        return MahjongHands.tile_pic_files[ind]

    def is_wind(self):
        return self.type == "wind"

    def is_dragon(self):
        return self.type == "dragon"

    def is_bamboo(self):
        return self.type == "bamboo"

    def is_char(self):
        return self.type == "character"

    def is_dot(self):
        return self.type == "dot"

    def get_dragon_type(self):
        if not self.is_dragon():
            return None
        return self.name[2]

    def get_wind_direction(self):
        if not self.is_wind():
            return None
        return self.name[1]

    def get_tile_number(self):
        if self.is_dragon() or self.is_wind():
            return None
        return self.name[1]

    def is_sequential_to(self, other):
        if self.type == other.type:
            if self.is_dragon() or self.is_wind():
                return False
            else:
                return int(self.get_tile_number()) == int(other.get_tile_number()) + 1
        return False

    def get_next_sequential_tile_name(self):
        if self.is_dragon() or self.is_wind():
            return None
        elif self.get_tile_number() == 9:
            return None
        return self.name[0] + str(int(self.get_tile_number()) + 1)

    @staticmethod
    def tile_name_to_next_tile_name(name):
        if name is None or name[:2] == "dr" or name[0] == "w":
            return None
        elif int(name[1]) == 9:
            return None
        else:
            return name[0] + str(int(name[1]) + 1)

    def __lt__(self, other):
        return self.name < other.name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return self.name.__hash__()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class Hand:
    def __init__(self):
        # Total set of tiles to be given from some other source
        self.concealed_tiles = []
        self.revealed_tiles = []
        self.declared_concealed_kongs = []
        self.num_suits_used = 0
        self.uses_bamboo = False
        self.uses_chars = False
        self.uses_dots = False
        self.num_winds = 0
        self.num_dragons = 0
        self.drawn_tile = None

    def _update_suits_and_honor_count(self):
        # Assume lists are sorted
        self.num_winds = 0
        self.num_dragons = 0
        self.num_suits_used = 0
        self.uses_bamboo = self.uses_dots = self.uses_chars = False
        all_tiles = self.concealed_tiles + self.revealed_tiles + flatten_list(self.declared_concealed_kongs)
        if self.drawn_tile:
            all_tiles += [self.drawn_tile]

        for tile in all_tiles:
            if tile.is_wind():
                self.num_winds += 1
            elif tile.is_dragon():
                self.num_dragons += 1
            elif not self.uses_bamboo and tile.is_bamboo():
                self.num_suits_used += 1
                self.uses_bamboo = True
            elif not self.uses_chars and tile.is_char():
                self.num_suits_used += 1
                self.uses_chars = True
            elif not self.uses_dots and tile.is_dot():
                self.num_suits_used += 1
                self.uses_dots = True

    def _hand_contains_concealed_chow_from_starting_tile(self, start_tile):
        next_tile_name = Tile.tile_name_to_next_tile_name(start_tile.name)
        if next_tile_name is None:
            return False
        next_next_tile_name = Tile.tile_name_to_next_tile_name(next_tile_name)
        tile_names = [t.name for t in self.concealed_tiles]
        return (next_tile_name in tile_names) and (next_next_tile_name in tile_names)

    def get_num_honor_tiles(self):
        return self.num_dragons + self.num_winds

    def set_drawn_tile(self, name):
        if Tile.is_valid_name(name):
            self.drawn_tile = Tile(name)
        self._update_hand()

    def get_num_tiles_in_hand(self):
        return len(flatten_list(self.declared_concealed_kongs)) + len(self.concealed_tiles) + len(self.revealed_tiles)

    def clear_hand(self):
        self.__init__()

    def _sort_hand(self):
        if len(self.concealed_tiles) > 1:
            self.concealed_tiles = sorted(self.concealed_tiles)
        if len(self.declared_concealed_kongs) >= 1:
            self.declared_concealed_kongs = sorted(self.declared_concealed_kongs)

    def _update_hand(self):
        self._sort_hand()
        self._update_suits_and_honor_count()

    def add_tile_to_hand(self, revealed, tile_name, more=False):
        # More is an optimization parameter. Set to true if you expect to continue adding pieces.
        # Setting the drawn tile will force an update, or call add tiles again with tile_name = None
        if self.get_num_tiles_in_hand() >= 16 or tile_name is None:
            self._update_hand()
            return
        if revealed:
            self.revealed_tiles += [Tile(tile_name)]
        else:
            self.concealed_tiles += [Tile(tile_name)]
        if not more:
            self._update_hand()

    def add_declared_concealed_kong_to_hand(self, tile_name):
        self.declared_concealed_kongs += [[Tile(tile_name), Tile(tile_name), Tile(tile_name), Tile(tile_name)]]
        self._update_suits_and_honor_count()

    def add_revealed_kong_to_hand(self, tile_name):
        # Not [Tile(tile_name)] * 4 because I don't want the same reference 4 times, but 4 different objects
        self.revealed_tiles += [Tile(tile_name), Tile(tile_name), Tile(tile_name), Tile(tile_name)]

    def is_fully_concealed(self):
        return not len(self.revealed_tiles)

    def _get_revealed_sets(self):
        # User is probably still typing
        if len(self.revealed_tiles) < 3:
            return [], [], []
        temp_set = self.revealed_tiles[:]
        revealed_chows = []
        revealed_pungs = []
        revealed_kongs = []
        if len(temp_set) == 3:
            t0, t1, t2 = temp_set[0], temp_set[1], temp_set[2]
            if t2.is_sequential_to(t1) and t1.is_sequential_to(t0):
                revealed_chows += [[t0, t1, t2]]
            else:
                revealed_pungs += [[t0, t1, t2]]
        elif len(temp_set) > 3:
            while len(temp_set) > 0:
                t0, t1, t2, t3 = temp_set[0], temp_set[1], temp_set[2], temp_set[3]
                if t0 == t1 and t1 == t2 and t2 == t3:
                    revealed_kongs += [[t0, t1, t2, t3]]
                    temp_set = temp_set[4:]
                elif t0 == t1 and t1 == t2 and t2 != t3:
                    revealed_pungs += [[t0, t1, t2]]
                    temp_set = temp_set[3:]
        return revealed_chows, revealed_pungs, revealed_kongs

    def get_num_revealed_sets(self, kind="all"):
        revealed_sets = self._get_revealed_sets()
        if kind == "all":
            return len(revealed_sets[0]) + len(revealed_sets[1]) + len(revealed_sets[2])
        elif kind == "chow":
            return len(revealed_sets[0])
        elif kind == "pung":
            return len(revealed_sets[1])
        elif kind == "kong":
            return len(revealed_sets[2])
        else:
            return -1


class PossibleWinningHand(Hand):
    def __init__(self, hand):
        super().__init__()
        self.revealed_tiles = hand.revealed_tiles[:]
        self.concealed_tiles = hand.concealed_tiles[:]
        self.declared_concealed_kongs = hand.declared_concealed_kongs[:]
        self.drawn_tile = hand.drawn_tile
        self.four_pair_base_dict = {}
        self.four_set_pair_hands = []
        self.special_hands = []
        self._update_hand()
        self._format_dict()
        self._construct_four_set_pair_hands()

    def _format_dict(self):
        rc, rp, rk = self._get_revealed_sets()
        self.four_pair_base_dict = {
            "revealed_chows": rc,
            "revealed_pungs": rp,
            "revealed_kongs": rk,
            "concealed_chows": [],
            "concealed_pungs": [],
            "concealed_kongs": [],
            "declared_concealed_kongs": self.declared_concealed_kongs,
            "pair": [],
            "discard": []
        }

    def _group_into_sets(self, tiles):
        if len(tiles) < 3:
            return [[]]
        t0 = tiles[0]
        t0_next = Tile(t0.get_next_sequential_tile_name())
        t0_third = Tile(t0_next.get_next_sequential_tile_name())
        sets_and_remainders = []
        if tiles.count(t0_next) >= 1 and tiles.count(t0_third) >= 1:
            ind_next = tiles.index(t0_next)
            ind_third = tiles.index(t0_third)
            remainder = tiles[:]
            remainder.pop(ind_third)
            remainder.pop(ind_next)
            remainder.pop(0)
            sets_and_remainders += [[
                [t0, tiles[ind_next], tiles[ind_third]],
                remainder
            ]]
        if tiles.count(t0) >= 3:
            temp_tiles = tiles[1:]
            ind_next = temp_tiles.index(t0)
            t0_1 = temp_tiles.pop(ind_next)
            ind_third = temp_tiles.index(t0)
            t0_2 = temp_tiles.pop(ind_third)
            sets_and_remainders += [[
                [t0, t0_1, t0_2],
                temp_tiles
            ]]
        if tiles.count(t0) == 4:
            temp_tiles = tiles[1:]
            ind_next = temp_tiles.index(t0)
            t0_1 = temp_tiles.pop(ind_next)
            ind_third = temp_tiles.index(t0)
            t0_2 = temp_tiles.pop(ind_third)
            ind_fourth = temp_tiles.index(t0)
            t0_3 = temp_tiles.pop(ind_fourth)
            sets_and_remainders += [[
                [t0, t0_1, t0_2, t0_3],
                temp_tiles
            ]]

        if not len(sets_and_remainders):
            return [[]]
        to_be_returned = []
        for starting in sets_and_remainders:
            for result in self._group_into_sets(starting[1]):
                to_be_returned += [[starting[0]] + result]
        return to_be_returned

    def _construct_four_set_pair_hands(self):
        self.four_set_pair_hands = []
        sets_remaining = 4 - self.get_num_revealed_sets("all") - len(self.declared_concealed_kongs)
        # Can we construct this many sets with the concealed tiles + drawn tile?
        all_tiles = self.concealed_tiles + [self.drawn_tile] if self.drawn_tile else self.concealed_tiles
        # Generate all possible pairs, and the remaining lists if you take out those pairs
        pairs_list = []
        tiles_minus_pairs = []
        for i in range(len(all_tiles)):
            temp_all_tiles = all_tiles[:]
            if all_tiles.count(all_tiles[i]) >= 2:
                t0 = temp_all_tiles.pop(i)
                if temp_all_tiles.index(all_tiles[i]) == i:
                    t1 = temp_all_tiles.pop(i)
                    pairs_list += [[t0, t1]]
                    tiles_minus_pairs += [temp_all_tiles]
        # Can't make a hand without a pair
        if len(pairs_list) == 0:
            return

        for i, leftover in enumerate(tiles_minus_pairs):
            s = self._group_into_sets(leftover)
            for combination in s:
                if len(combination) == sets_remaining:
                    tile_to_discard = list(set(leftover) - set(flatten_list(combination)))
                    temp_dict = deepcopy(self.four_pair_base_dict)
                    temp_dict["discard"] += tile_to_discard
                    for complete_set in combination:
                        if len(complete_set) == 4:
                            temp_dict["concealed_kongs"] += [complete_set]
                        elif complete_set[0] == complete_set[1]:
                            temp_dict["concealed_pungs"] += [complete_set]
                        else:
                            temp_dict["concealed_chows"] += [complete_set]
                    temp_dict["pair"] = pairs_list[i][:]
                    self.four_set_pair_hands += [temp_dict]
        return


